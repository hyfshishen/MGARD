#include <cuda.h>
#include <cuda_runtime.h>

struct PrivateTimingGPU;

class TimingGPU
{
    private:
        PrivateTimingGPU *privateTimingGPU;

    public:

        TimingGPU();

        ~TimingGPU();

        void StartCounter();
        void StartCounterFlags();

        float GetCounter();

}; // TimingGPU class

struct PrivateTimingGPU {
    cudaEvent_t start;
    cudaEvent_t stop;
};

// default constructor
TimingGPU::TimingGPU() { privateTimingGPU = new PrivateTimingGPU;  }

// default destructor
TimingGPU::~TimingGPU() { }

void TimingGPU::StartCounter()
{
    cudaEventCreate(&((*privateTimingGPU).start));
    cudaEventCreate(&((*privateTimingGPU).stop));
    cudaEventRecord((*privateTimingGPU).start,0);
}

void TimingGPU::StartCounterFlags()
{
    int eventflags = cudaEventBlockingSync;

    cudaEventCreateWithFlags(&((*privateTimingGPU).start),eventflags);
    cudaEventCreateWithFlags(&((*privateTimingGPU).stop),eventflags);
    cudaEventRecord((*privateTimingGPU).start,0);
}

// Gets the counter in ms
float TimingGPU::GetCounter()
{
    float time;
    cudaEventRecord((*privateTimingGPU).stop, 0);
    cudaEventSynchronize((*privateTimingGPU).stop);
    cudaEventElapsedTime(&time,(*privateTimingGPU).start,(*privateTimingGPU).stop);
    return time;
}