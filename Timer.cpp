#include "stdafx.h"
#include "Timer.h"


Timer::Timer()
	: counter(0)
{

}

void Timer::createTimePoint()
{
	times[std::to_string(counter)] = std::chrono::steady_clock::now();

	counter++;
}

void Timer::createTimePoint(const std::string& pointName)
{
	times[pointName] = std::chrono::steady_clock::now();

	counter++;
}

long long Timer::getTimeDifferenceMs(const std::string& pointOne, const std::string& pointTwo) const
{
	auto timeOne = times.at(pointOne);
	auto timeTwo = times.at(pointTwo);

	return std::chrono::duration_cast<std::chrono::microseconds>(timeTwo - timeOne).count();
}

long long Timer::getTimeDifferenceSec(const std::string& pointOne, const std::string& pointTwo) const
{
	auto timeOne = times.at(pointOne);
	auto timeTwo = times.at(pointTwo);

	return std::chrono::duration_cast<std::chrono::seconds>(timeTwo - timeOne).count();
}

void Timer::printTimeDifferenceMs(const std::string& pointOne, const std::string& pointTwo) const
{
	std::cout <<"Time elased(ms) between "<<pointOne << " and " << pointTwo <<
		" is: " << getTimeDifferenceMs(pointOne, pointTwo) << "\n";
}

void Timer::printTimeDifferenceSec(const std::string& pointOne, const std::string& pointTwo) const
{
	std::cout << "Time elased(sec) between " << pointOne << " and " << pointTwo <<
		" is: " << getTimeDifferenceSec(pointOne, pointTwo) << "\n";
}

Timer::~Timer()
{

}
