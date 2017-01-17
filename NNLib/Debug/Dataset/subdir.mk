################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Dataset/Dataset.cpp 

OBJS += \
./Dataset/Dataset.o 

CPP_DEPS += \
./Dataset/Dataset.d 


# Each subdirectory must supply rules for building sources it contributes
Dataset/%.o: ../Dataset/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


