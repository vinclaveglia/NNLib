################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../FeedForward/FeedForwardLayer.cpp \
../FeedForward/Layer2.cpp 

OBJS += \
./FeedForward/FeedForwardLayer.o \
./FeedForward/Layer2.o 

CPP_DEPS += \
./FeedForward/FeedForwardLayer.d \
./FeedForward/Layer2.d 


# Each subdirectory must supply rules for building sources it contributes
FeedForward/%.o: ../FeedForward/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


