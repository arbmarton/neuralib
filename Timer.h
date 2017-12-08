#pragma once

#include <map>
#include <chrono>
#include <string>
#include <iostream>

class Timer
{
public:
	Timer();

	void createTimePoint();
	void createTimePoint(const std::string& pointName);

	long long getTimeDifferenceMs(const std::string& pointOne, const std::string& pointTwo) const;
	long long getTimeDifferenceSec(const std::string& pointOne, const std::string& pointTwo) const;

	void printTimeDifferenceMs(const std::string& pointOne, const std::string& pointTwo) const;
	void printTimeDifferenceSec(const std::string& pointOne, const std::string& pointTwo) const;

	~Timer();
private:
	int counter;
	std::map<std::string, std::chrono::steady_clock::time_point> times;
};

