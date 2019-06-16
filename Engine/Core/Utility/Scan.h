#pragma once

#define SCAN_BLOCK_SIZE 128

class Scan
{
public:
	static Scan* create(int buf_size, bool inOrder = true);
	~Scan();

	int ExclusiveScan(int* dst, int* src, int size);

	int ExclusiveScan(int* src, int size);

	int getBufferSize() { return m_size; }

private:
	Scan();
	void allocateBuffer(int size);

	int m_size = 0;

	int* m_sum = nullptr;
	int* m_buffer = nullptr;
};

