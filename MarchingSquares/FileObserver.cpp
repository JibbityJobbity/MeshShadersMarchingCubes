#include "stdafx.h"
#include "FileObserver.h"

#include <iostream>
using namespace std;	

FileObserver::FileObserver(string path, chrono::duration<uint64_t, std::milli> duration) : m_duration(duration), m_running(true), Updated(false) {
	m_path = path;
	m_thread = std::thread(&FileObserver::observe, this);
}

FileObserver::FileObserver(FileObserver& other) {
	m_path = other.m_path;
	m_duration = other.m_duration;
	m_running = other.m_running;
	m_thread = std::thread(&FileObserver::observe, this);
}

FileObserver::FileObserver(FileObserver&& other) {
	m_path = other.m_path;
	m_duration = other.m_duration;
	m_running = other.m_running;
	m_thread = std::move(other.m_thread);
}

void FileObserver::Reset() {
	Updated = false;
}

void FileObserver::observe() {
	//this_thread::sleep_for(chrono::milliseconds(5000));
	auto prevLastWrite = filesystem::last_write_time(m_path);
	while (m_running) {
		this_thread::sleep_for(chrono::milliseconds(m_duration));
		auto lastWrite = filesystem::last_write_time(m_path);
		auto difference = lastWrite - prevLastWrite;
		if (difference > std::chrono::duration<uint64_t, std::milli>(0)) {
			Updated = true;	
			prevLastWrite = lastWrite;
		}
	}
}

void FileObserver::Join() {
	m_thread.join();
}

void FileObserver::Stop() {
	m_running = false;
}
