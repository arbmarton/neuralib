#pragma once

#include <map>
#include <chrono>
#include <string>

class Timer
{
public:
	Timer();

	void createTimePoint();
	void createTimePoint(const std::string& pointName);

	long long getTimeDifferenceMs(const std::string& pointOne, const std::string& pointTwo) const;

	~Timer();
private:
	int counter;
	std::map<std::string, std::chrono::steady_clock::time_point> times;
};

