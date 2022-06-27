#ifndef FILE_OBSERVER_H
#define FILE_OBSERVER_H

#include <thread>
#include <string>
#include <filesystem>
#include <chrono>

class FileObserver {
public:
	volatile bool Updated;
	FileObserver(std::string path, std::chrono::duration<uint64_t, std::milli> duration);
	void Reset();
	void Stop();
	void Join();
protected:
	std::string m_path;
	std::chrono::duration<uint64_t, std::milli> m_duration;
	std::thread m_thread;
	volatile bool m_running;
	void observe();
private:
	FileObserver(FileObserver&);
	FileObserver(FileObserver&&);
};

#endif
