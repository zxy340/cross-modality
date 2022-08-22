#include <Windows.h>
#include <Ole2.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <io.h>
#include <direct.h>

#include <Kinect.h>

#define RGBWidth 1920
#define RGBHeight 1080
#define DepWidth 512
#define DepHeight 424
#define times 500

// Kinect vairables
IKinectSensor* sensor;               // Kinect sensor
IMultiSourceFrameReader* reader;     // Kinect data source
ICoordinateMapper* mapper;           // Converts between depth, color, and 3d coordinates
std::vector<std::vector <int> > timestamp(times, std::vector<int>(7, 0));   // Record the timestamp of the frame
SYSTEMTIME st;
byte RGBdata[RGBWidth * RGBHeight * 4];  // BGRA array containing the texture data
UINT16 Depdata[DepWidth * DepHeight * 3];  // BGRA array containing the texture data

ColorSpacePoint depth2rgb[DepWidth * DepHeight];     // Maps depth pixels to RGB pixels
CameraSpacePoint depth2xyz[DepWidth * DepHeight];    // Maps depth pixels to 3d coordinates
DepthSpacePoint rgb2depth[RGBWidth * RGBHeight];     // Maps RGB pixels to depth pixels
CameraSpacePoint rgb2xyz[RGBWidth * RGBHeight];      // Maps RGB pixels to 3d coordinates

bool initKinect() {
    if (FAILED(GetDefaultKinectSensor(&sensor))) {
        return false;
    }
    if (sensor) {
        sensor->get_CoordinateMapper(&mapper);

        sensor->Open();
        sensor->OpenMultiSourceFrameReader(
            FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color,
            &reader);
        return reader;
    }
    else {
        return false;
    }
}

void getDepthData(IMultiSourceFrame* frame) {
    IDepthFrame* depthframe;
    IDepthFrameReference* frameref = NULL;
    frame->get_DepthFrameReference(&frameref);
    frameref->AcquireFrame(&depthframe);
    if (frameref) frameref->Release();
    if (!depthframe) return;

    depthframe->CopyFrameDataToArray(DepWidth * DepHeight, Depdata);

    // Process depth frame data...
    unsigned int sz;
    unsigned short* buf;
    if (SUCCEEDED(depthframe->AccessUnderlyingBuffer(&sz, &buf))) {
        if (SUCCEEDED(mapper->MapDepthFrameToCameraSpace(
            DepWidth * DepHeight, buf,
            DepWidth * DepHeight, depth2xyz))) {
            // std::cout << "depth2xyz successfully got!" << std::endl;
        }          // Output CameraSpacePoint array and size
        if (SUCCEEDED(mapper->MapDepthFrameToColorSpace(
            DepWidth * DepHeight, buf,
            DepWidth * DepHeight, depth2rgb))) {
            // std::cout << "depth2rgb successfully got!" << std::endl;
        }          // Output ColorSpacePoint array and size
        if (SUCCEEDED(mapper->MapColorFrameToCameraSpace(
            DepWidth * DepHeight, buf,
            RGBWidth * RGBHeight, rgb2xyz))) {
            // std::cout << "rgb2xyz successfully got!" << std::endl;
        }                           // Output CameraSpacePoint array and size
        if (SUCCEEDED(mapper->MapColorFrameToDepthSpace(
            DepWidth * DepHeight, buf,
            RGBWidth * RGBHeight, rgb2depth))) {
            // std::cout << "rgb2depth successfully got!" << std::endl;
        }                         // Output ColorSpacePoint array and size
    };
    if (depthframe) depthframe->Release();
}

void getColorData(IMultiSourceFrame* frame) {
    IColorFrame* colorframe;
    IColorFrameReference* frameref = NULL;
    frame->get_ColorFrameReference(&frameref);
    frameref->AcquireFrame(&colorframe);
    if (frameref) frameref->Release();
    if (!colorframe) return;
    colorframe->CopyConvertedFrameDataToArray(RGBWidth * RGBHeight * 4, RGBdata, ColorImageFormat_Rgba);

    if (colorframe) colorframe->Release();
}

void writeData() {
    //std::cout << "start writing data!" << std::endl;
    FILE* pFile1;
    if ((pFile1 = fopen("./data/RandomHand/1/Kinect/Depdata.txt", "ab")) == NULL) {
        printf("cant open the file");
        exit(0);
    }
    fwrite(Depdata, sizeof(UINT16), DepWidth * DepHeight * 3, pFile1);
    fclose(pFile1);
    // std::cout << "Depdata.txt succefully writed!" << std::endl;
    FILE* pFile2;
    if ((pFile2 = fopen("./data/RandomHand/1/Kinect/Depth2xyz.txt", "ab")) == NULL) {
        printf("cant open the file");
        exit(0);
    }
    FLOAT* Depth2xyz = new FLOAT[DepWidth * DepHeight * 3];
    for (int i = 0; i < DepWidth * DepHeight; i++) {
        Depth2xyz[3 * i + 0] = depth2xyz[i].X;
        Depth2xyz[3 * i + 1] = depth2xyz[i].Y;
        Depth2xyz[3 * i + 2] = depth2xyz[i].Z;
    }
    fwrite(Depth2xyz, sizeof(FLOAT), DepWidth * DepHeight * 3, pFile2);
    fclose(pFile2);
    // std::cout << "Depth2xyz.txt succefully writed!" << std::endl;
    FILE* pFile3;
    if ((pFile3 = fopen("./data/RandomHand/1/Kinect/Depth2rgb.txt", "ab")) == NULL) {
        printf("cant open the file");
        exit(0);
    }
    FLOAT* Depth2rgb = new FLOAT[DepWidth * DepHeight * 2];
    for (int i = 0; i < DepWidth * DepHeight; i++) {
        Depth2rgb[2 * i + 0] = depth2rgb[i].X;
        Depth2rgb[2 * i + 1] = depth2rgb[i].Y;
    }
    fwrite(Depth2rgb, sizeof(FLOAT), DepWidth * DepHeight * 2, pFile3);
    fclose(pFile3);
    // std::cout << "Depth2rgb.txt succefully writed!" << std::endl;
    FILE* pFile4;
    if ((pFile4 = fopen("./data/RandomHand/1/Kinect/RGBdata.txt", "ab")) == NULL) {
        printf("cant open the file");
        exit(0);
    }
    fwrite(RGBdata, sizeof(byte), RGBWidth * RGBHeight * 4, pFile4);
    fclose(pFile4);
    // std::cout << "RGBdata.txt succefully writed!" << std::endl;
    FILE* pFile5;
    if ((pFile5 = fopen("./data/RandomHand/1/Kinect/Rgb2depth.txt", "ab")) == NULL) {
        printf("cant open the file");
        exit(0);
    }
    FLOAT* Rgb2depth = new FLOAT[RGBWidth * RGBHeight * 2];
    for (int i = 0; i < RGBWidth * RGBHeight; i++) {
        Rgb2depth[2 * i + 0] = rgb2depth[i].X;
        Rgb2depth[2 * i + 1] = rgb2depth[i].Y;
    }
    fwrite(Rgb2depth, sizeof(FLOAT), RGBWidth * RGBHeight * 2, pFile5);
    fclose(pFile5);
    // std::cout << "Rgb2depth.txt succefully writed!" << std::endl;
    FILE* pFile6;
    if ((pFile6 = fopen("./data/RandomHand/1/Kinect/Rgb2xyz.txt", "ab")) == NULL) {
        printf("cant open the file");
        exit(0);
    }
    FLOAT* Rgb2xyz = new FLOAT[RGBWidth * RGBHeight * 3];
    for (int i = 0; i < RGBWidth * RGBHeight; i++) {
        Rgb2xyz[3 * i + 0] = rgb2xyz[i].X;
        Rgb2xyz[3 * i + 1] = rgb2xyz[i].Y;
        Rgb2xyz[3 * i + 2] = rgb2xyz[i].Z;
    }
    fwrite(Rgb2xyz, sizeof(FLOAT), RGBWidth * RGBHeight * 3, pFile6);
    fclose(pFile6);
    // std::cout << "Rgb2xyz.txt succefully writed!" << std::endl;
}

int getKinectData(int time) {
    IMultiSourceFrame* frame = NULL;
    if (SUCCEEDED(reader->AcquireLatestFrame(&frame))) {
        GetLocalTime(&st);
        std::cout << st.wYear << "." << st.wMonth << "." << st.wDay << "."
            << st.wHour << "." << st.wMinute << "." << st.wSecond << "." << st.wMilliseconds << std::endl;
        timestamp[time][0] = (int)st.wYear;
        timestamp[time][1] = (int)st.wMonth;
        timestamp[time][2] = (int)st.wDay;
        timestamp[time][3] = (int)st.wHour;
        timestamp[time][4] = (int)st.wMinute;
        timestamp[time][5] = (int)st.wSecond;
        timestamp[time][6] = (int)st.wMilliseconds;
        getDepthData(frame);
        getColorData(frame);
        writeData();
        time++;
    }
    if (frame) frame->Release();
    return time;
}


int main() {
    if (!initKinect()) return 1;
    std::cout << "start in 10 seconds" << std::endl;
    Sleep(2000);
    std::cout << "start in 8 seconds" << std::endl;
    Sleep(2000);
    std::cout << "start in 6 seconds" << std::endl;
    Sleep(2000);
    std::cout << "start in 4 seconds" << std::endl;
    Sleep(2000);
    std::cout << "start in 2 seconds" << std::endl;
    Sleep(2000);
    int time = 0;
    while (time < times) {
        time = getKinectData(time);
        Sleep(50);
    }
    std::ofstream myfile1("./data/RandomHand/1/Kinect/timestamp.txt");
    if (myfile1.is_open())
    {
        int time = 0;
        while (time < times) {
            for (int i = 0; i < 7; i++) {
                myfile1 << timestamp[time][i] << " ";
            }
            myfile1 << "\n";
            time++;
        }
        myfile1.close();
    }

    return 0;
}