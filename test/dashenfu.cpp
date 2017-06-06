#include "ImageConsProd.hpp"

#include <thread>

int main(int argc, char** argv)
{
	
	// start threads
	ImageConsProd image_cons_prod;
	std::thread t1(&ImageConsProd::ImageProducer, image_cons_prod);
	std::thread t2(&ImageConsProd::ImageConsumer, image_cons_prod);

	t1.join();
	t2.join();

}
