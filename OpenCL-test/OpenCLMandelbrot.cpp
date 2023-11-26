#include <CL/cl.h>

#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>

int align(int x, int y) {
    return (x + y - 1) / y * y;
}

void invoke_kernel(cl_kernel kernel, cl_command_queue queue, cl_mem buff, cl_uint* result,
    float x, float y, float mag, int w, int h, float iterations) {
    clSetKernelArg(kernel, 0, sizeof(float), &x);
    clSetKernelArg(kernel, 1, sizeof(float), &y);
    clSetKernelArg(kernel, 2, sizeof(float), &mag);
    clSetKernelArg(kernel, 3, sizeof(float), &iterations);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &w);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &h);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &buff);
    clSetKernelArg(kernel, 7, sizeof(cl_int), &w);

    size_t local_size[2] = { 256, 1 };
    size_t global_size[2] = { align(w, local_size[0]),
           align(h, local_size[1]) };

    // запускаем двумерную задачу
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
        global_size, local_size, 0, NULL, NULL);

    // читаем результат
    clEnqueueReadBuffer(queue, buff, CL_TRUE, 0,
        sizeof(int) * w * h, result, 0, NULL, NULL);

    // ждём завершения всех операций
    clFinish(queue);

}

cl_device_id create_device() {
    cl_platform_id platform;
    cl_device_id dev;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    return dev;
}

std::string get_program_text() {
    std::ifstream t("mandelbrot.cl");
    return std::string((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());
}

cl_program build_program(cl_context ctx, cl_device_id dev) {
    int err;
    std::string src = get_program_text();
    const char* src_text = src.data();
    size_t src_length = src.size();
    cl_program program = clCreateProgramWithSource(ctx, 1, &src_text, &src_length, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    return program;
}

void save_ppm(const cl_uint* p, int w, int h) {
    std::ofstream file("mandelbrot.ppm", std::ios::binary);
    file << "P6\n" << w << " " << h << "\n255\n";
    for (int y = 0; y < h; ++y) {
        const cl_uint* line = p + w * y;
        for (int x = 0; x < w; ++x) {
            file.write((const char*)(line + x), 3);
        }
    }
}

int main() {
    static const int res_w = 1200;
    static const int res_h = 640;

    cl_int err;
    cl_device_id device = create_device();
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    cl_program program = build_program(context, device);
    cl_kernel kernel = clCreateKernel(program, "draw_mandelbrot", &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    cl_mem buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint) * res_w * res_h, NULL, NULL);

    std::vector<cl_uint> pixels(res_w * res_h);
    invoke_kernel(kernel, queue, buff, pixels.data(), -.5f, 0, 4.5f, res_w, res_h, 50);
    save_ppm(pixels.data(), res_w, res_h);

    /* Deallocate resources */
    clReleaseKernel(kernel);
    clReleaseMemObject(buff);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}
