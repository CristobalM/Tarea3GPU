#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <chrono>
#include <GL/glew.h>
#include <CL/opencl.h>
#include <GL/freeglut.h>
#include <GL/glx.h>

int sample_rate = 10000;
auto start = std::chrono::high_resolution_clock::now();
auto finish = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> timelapse;

// OpenGL variables for window and orthogonal matrix

GLfloat angle = 0.0f;

int window_width = 1400;
int window_height = 900;
int window_pos_x = 250;
int window_pos_y = 50;
int refresh_fequency = 1000.0/30.0; // In milliseconds (1000.0/1.0 = once per second)

double zoom = 1.0;
double ortho_top = 1.0;
double ortho_left = -1.0;
double ortho_right = 1.0;
double ortho_bottom = -1.0;

char title[] = "High Life, variant of Game of Life (OpenCL version)";

bool full_screen_mode = true;

clock_t life_clock = 0;

GLuint frame_buffer_name = 0;
GLuint rendered_texture, rendered_texture_out;


// OpenCL Variables

cl_context context;
cl_context_properties properties[7];

cl_device_id device;
cl_platform_id platform;

cl_command_queue command_quque;

cl_kernel kernel;
cl_program program;

cl_mem dev_hl_image;
cl_mem dev_hl_map_in;
cl_mem dev_hl_map_out;

size_t size;
cl_int err;

char *build_log;

clGetGLContextInfoKHR_fn myGetGLContextInfoKHR;


// HighLife Variables

char *hl_map = NULL;
char *hl_tmap = NULL;
int hl_map_width = 2000;
int hl_map_height = 1400;
long long int hl_generation = 0;


// Work Size Variables

size_t elements_size[2];
size_t global_work_size[2];
size_t local_work_size[2];
size_t param_data_bytes;
size_t kernel_length;
size_t build_log_size_ret;


// OpenGL Functions

void initGL(int argc, char *argv[]);
void startGL();
void display();
void displayTimer(int value);
void generationTimer(int value);
void reshape(GLsizei width, GLsizei height);


// OpenCL Kernel

const char *kernel_source = "highLifeEngine.cl";
char *source_string = NULL;


// HighLife Functions 

void hlMapClear();
void hlMapDump();
void hlMapGenerate();
void hlMapRandFill();
inline char hlCellNext(int x, int y);
inline char hlCellGet(int x, int y);
inline void hlCellSet(int x, int y, char value);
inline void hlCellDraw(int x, int y);


// Utilities to make life easier

int loadProgramSource(const char *filename, char **p_source_string, size_t *length);
void clean();
void die(cl_int err, const char* str);


// Utilities to make life easier

int loadProgramSource(const char *filename, char **p_source_string, size_t *length) {
    FILE *file;
    size_t source_length;

    file = fopen(filename, "rb");
    if(file == 0) {
        return 1;
    }

    fseek(file, 0, SEEK_END);
    source_length = ftell(file);
    fseek(file, 0, SEEK_SET);

    *p_source_string = (char *)malloc(source_length + 1);
    if(fread(*p_source_string, source_length, 1, file) != 1) {
        fclose(file);
        free(*p_source_string);
        return 1;
    }

    fclose(file);
    *length = source_length;
    (*p_source_string)[source_length] = '\0';

    return 0;
}

void clean() {
    if(source_string) free(source_string);
    if(kernel) free(kernel);
    if(program) free(program);
    if(command_quque) free(command_quque);
    if(context) free(context);
    if(device) free(device);
    if(platform) free(platform);
    if(dev_hl_map_in) free(dev_hl_map_in);
    if(dev_hl_map_out) free(dev_hl_map_out);

    free(hl_map);
    free(hl_tmap);
}

void die(cl_int err, const char* str) {
    if(err != CL_SUCCESS) {
        fprintf(stderr, "Error code %d: %s\n", err, str);
        clean();
        exit(1);
    }
}


// OpenGL Functions

void initGL(int argc, char *argv[]) {
    // Basic init
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition(window_pos_x, window_pos_y);
    glutCreateWindow(title);

    // Bind functions and clear color
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Textures
    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &rendered_texture);
    glBindTexture(GL_TEXTURE_2D, rendered_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, hl_map_width, hl_map_height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);

    glFinish();
}

void startGL() {
    glutTimerFunc(0, displayTimer, 0);
    glutTimerFunc(0, generationTimer, 0);
    glutMainLoop();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glOrtho(ortho_left, ortho_right, ortho_bottom, ortho_top, -10, 10);
    glScalef(zoom, zoom, 1.0);

    glEnable(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, hl_map_width, hl_map_height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, hl_map);
    glBindTexture(GL_TEXTURE_2D, rendered_texture);

    glBegin(GL_QUADS);
    
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.0);
    glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0, 0.0);
    glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, 0.0);
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, 0.0);
    
    glEnd();

    glutSwapBuffers();
}

void displayTimer(int value) {
    glutPostRedisplay();
    glutTimerFunc(refresh_fequency, displayTimer, 0);
}

void generationTimer(int value) {
    err = clEnqueueAcquireGLObjects(command_quque, 1, &dev_hl_image, 0, 0, 0);
    die(err, "clEnqueueAcquireGLObjects");

    clEnqueueNDRangeKernel(command_quque, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    die(err, "clEnqueueNDRangeKernel");

    clFinish(command_quque);

    err = clEnqueueCopyBuffer(command_quque, dev_hl_map_out, dev_hl_map_in, 0, 0, sizeof(unsigned char) * global_work_size[0] * global_work_size[1], 0, NULL, NULL);
    die(err, "clEnqueueCopyBuffer");

    err = clEnqueueReleaseGLObjects(command_quque, 1,  &dev_hl_image, 0, 0, NULL);
    die(err, "clEnqueueReleaseGLObjects");
    
    hl_generation++;

    if(hl_generation % sample_rate == 0) {
        clock_t now = clock();
        finish = std::chrono::high_resolution_clock::now();
        timelapse = finish - start;
        double fps = (double)sample_rate / ( (now - life_clock) / CLOCKS_PER_SEC );
        printf("generation : %d\n", hl_generation);
        printf("fps = %lf\n", fps);
        printf("timelapse between generation %d and %d: %lf miliseconds (%lf seconds)\n", hl_generation - sample_rate, hl_generation, timelapse.count() * 1000, timelapse.count());
        life_clock = now;
        start = std::chrono::high_resolution_clock::now();
    }

    glutTimerFunc(0, generationTimer, 0);
}

void reshape(GLsizei width, GLsizei height) {
    // Compute aspect ratio of the new window
    if (height == 0) {
        height = 1;
    }
    GLfloat aspect = (GLfloat)width / (GLfloat)height;

    // Set the viewport to cover the new window
    glViewport(0, 0, width, height);

    // Set the aspect ratio of the clipping area to match the viewport
    glMatrixMode(GL_PROJECTION);  
    glLoadIdentity();    

    if (width >= height) {
        gluOrtho2D(-1.0 * aspect, 1.0 * aspect, -1.0, 1.0);
    } else {
        gluOrtho2D(-1.0, 1.0, -1.0 / aspect, 1.0 / aspect);
    }
}


// HighLife Functions 

void hlMapClear() {
    memset(hl_map, 0, hl_map_width * hl_map_height * sizeof(char));
}

void hlMapDump() {
    for(int j = 0; j < hl_map_width; ++j) {
        for(int i = 0; i < hl_map_height; ++i) {
            printf("%c", hlCellGet(j, i)?'.':' ');
        }
        printf("\n");
    }
}

void hlMapGenerate() {
    for(int j = 0; j < hl_map_width; ++j) {
        for(int i = 0; i < hl_map_height; ++i) {
            hl_tmap[i * hl_map_width + j] = hlCellNext(j, i);
        }
    }
    memcpy(hl_map, hl_tmap, hl_map_width * hl_map_height * sizeof(char));
}

void hlMapRandFill() {
    unsigned seed = time(0);
    srand(seed);

    for(int j = 0; j < hl_map_width; ++j) {
        for(int  i = 0; i < hl_map_height; ++i) {
            hl_map[i * hl_map_width + j] = (rand() & 0x1);
        }
    }
}

inline char hlCellNext(int x, int y) {
    int cell_count = 0;

    if(hlCellGet(x-1, y-1)) ++cell_count;
    if(hlCellGet(x-1, y  )) ++cell_count;
    if(hlCellGet(x-1, y+1)) ++cell_count;
    if(hlCellGet(x  , y-1)) ++cell_count;
    if(hlCellGet(x  , y+1)) ++cell_count;
    if(hlCellGet(x+1, y-1)) ++cell_count;
    if(hlCellGet(x+1, y  )) ++cell_count;
    if(hlCellGet(x+1, y+1)) ++cell_count;

    if(hlCellGet(x, y) == 1 && !(cell_count == 3 || cell_count == 2))
        return 0;
    else if(hlCellGet(x, y) == 0 && (cell_count == 3 || cell_count == 6))
        return 1;
    else
        return hlCellGet(x, y);
}

inline char hlCellGet(int x, int y) {
    if(x < 0 || y < 0 || x > hl_map_width || y > hl_map_height){
        return false;
    } else {
        return hl_map[y * hl_map_width + x];
    }
}

inline void hlCellSet(int x, int y, char value) {
    if(x < 0 || y < 0 || x > hl_map_width || y > hl_map_height) {
        return;
    } else {
        hl_map[y * hl_map_width + x] = value;
    }
}

inline void hlCellDraw(int x, int y) {
    float posX = -1.0 + 2.0 * x / hl_map_width;
    float posY = -1.0 + 2.0 * y / hl_map_height;
    float posX2 = posX + 2.0 / hl_map_width;
    float posY2 = posY + 2.0 / hl_map_height;

    glColor3f(1.0f, 1.0f, 0.0f); // Green
    glRectf(posX, posY, posX2, posY2);
}


// Main for HighLife, where it all comes together.

int main(int argc, char *argv[]) {
    // Init OpenGL
    initGL(argc, argv);

    elements_size[0] = hl_map_width; 
    elements_size[1] = hl_map_height;

    local_work_size[0] = 32;
    local_work_size[1] = 32;

    global_work_size[0] = (size_t)ceil((double)elements_size[0] / local_work_size[0]) * local_work_size[0];
    global_work_size[1] = (size_t)ceil((double)elements_size[1] / local_work_size[1]) * local_work_size[1];

    printf("global_work_size[0]=%u, local_work_size[0]=%u, elements_size[0]=%u, work_groups_x=%u\n",
            global_work_size[0], local_work_size[0], elements_size[0], global_work_size[0] / local_work_size[0]);
    printf("global_work_size[1]=%u, local_work_size[1]=%u, elements_size[1]=%u, work_groups_y=%u\n",
            global_work_size[1], local_work_size[1], elements_size[1], global_work_size[1] / local_work_size[1]);


    // Host Memory Allocation
    hl_map      = (char *)malloc(sizeof(cl_char) * global_work_size[0] * global_work_size[1]);
    hl_tmap     = (char *)malloc(sizeof(cl_char) * global_work_size[0] * global_work_size[1]);
    

    // Init HighLife Map
    hlMapRandFill();


    // OpenCL Configurations
    err = clGetPlatformIDs(1, &platform, NULL);
    die(err, "clGetPlatformIds");

    properties[0] = CL_GL_CONTEXT_KHR;  properties[1] = (cl_context_properties)glXGetCurrentContext();
    properties[2] = CL_GLX_DISPLAY_KHR; properties[3] = (cl_context_properties)glXGetCurrentDisplay();
    properties[4] = CL_CONTEXT_PLATFORM;  properties[5] = (cl_context_properties)platform;
    properties[6] = 0;

    myGetGLContextInfoKHR = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");

    myGetGLContextInfoKHR(properties, CL_DEVICES_FOR_GL_CONTEXT_KHR, sizeof(cl_device_id), &device, &size);

    context =  clCreateContext(properties, 1, &device, NULL, NULL, &err);
    die(err, "clCreateContext");

    printf("dev=%u, ctx=%u\n", device, context);

    command_quque = clCreateCommandQueue(context, device, 0, &err);
    die(err, "clCreateCommandQueue");


    // OpenCL Buffers Init
    dev_hl_image = clCreateFromGLTexture2D(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, rendered_texture, &err);
    die(err, "clCreateBuffer dev_hl_image");

    dev_hl_map_in = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * global_work_size[0] * global_work_size[1] , NULL, &err);
    die(err, "clCreateBuffer dev_hl_map_in");

    dev_hl_map_out = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_char) * global_work_size[0] * global_work_size[1], NULL, &err);
    die(err, "clCreateBuffer dev_hl_map_out");


    // Loading and Building Program
    loadProgramSource(kernel_source, &source_string, &kernel_length);

    program = clCreateProgramWithSource(context, 1, (const char **)&source_string, &kernel_length, &err);
    die(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS) {
        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size_ret);

        build_log = (char *)malloc(build_log_size_ret);

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size_ret, build_log, NULL);

        printf("%s\n", build_log);
        exit(1);
    }


    // Creating Kernel and Setting Arguments
    kernel = clCreateKernel(program, "highLifeEngine", &err);
    die(err, "clCreateKernel highLifeEngine");

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dev_hl_map_in);
    die(err, "clSetKernelArg 0");
    err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dev_hl_map_out);
    die(err, "clSetKernelArg 1");
    err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&dev_hl_image);
    die(err, "clSetKernelArg 1");
    err  = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&elements_size[0]);
    die(err, "clSetKernelArg 3");
    err  = clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&elements_size[1]);
    die(err, "clSetKernelArg 4");

    err  = clEnqueueWriteBuffer(command_quque, dev_hl_map_in, CL_FALSE, 0, sizeof(cl_char) * global_work_size[0] * global_work_size[1] , hl_map, 0, NULL, NULL);
    die(err, "clEnqueueWriteBuffer");

    startGL();


    // END
    clean();
    return 0;
}
