#include "include_files.h"

void ImageProcessing(CPFloatImage& F, float* r_in, float* b_in, float* g_in,
					 float* r_out, float* b_out, float* g_out)
{
	parallel_for(blocked_range2d<int, int>(0, F.h, 0, F.w),
				 [&](const blocked_range2d<int>& r)
	{
		int baseX	= r.cols().begin();
		int baseY	= r.rows().begin();
		int numRows	= r.rows().end() - baseY;
		int rowSpan	= r.cols().end() - baseX;

		float *iRedBase	  = r_in  + (baseY * F.w) + baseX;
		float *iGreenBase = g_in  + (baseY * F.w) + baseX;
		float *iBlueBase  = b_in  + (baseY * F.w) + baseX;
		float *oRedBase	  = r_out + (baseY * F.w) + baseX;
		float *oGreenBase = g_out + (baseY * F.w) + baseX;
		float *oBlueBase  = b_out + (baseY * F.w) + baseX;

		for (auto i = 0; i < numRows; i++, 
			iRedBase += F.w, iGreenBase += F.w, iBlueBase += F.w, 
			oRedBase += F.w, oGreenBase += F.w, oBlueBase += F.w)
		{
			float *iRedPtr	 = iRedBase;
			float *iGreenPtr = iGreenBase;
			float *iBluePtr  = iBlueBase;
			float *oRedPtr   = oRedBase;
			float *oGreenPtr = oGreenBase;
			float *oBluePtr  = oBlueBase;

			for (auto j = 0; j < rowSpan; j++, 
				iRedPtr++, iGreenPtr++, iBluePtr++,
				oRedPtr++, oGreenPtr++, oBluePtr++)
			{
				*oRedPtr   = 0.4124f * *iRedPtr + 0.3576f * *iGreenPtr + 0.1805f * *iBluePtr;
				*oGreenPtr = 0.2126f * *iRedPtr + 0.7152f * *iGreenPtr + 0.0722f * *iBluePtr;
				*oBluePtr  = 0.0193f * *iRedPtr + 0.1192f * *iGreenPtr + 0.9505f * *iBluePtr;
			}
		}
	}
	);
}

void ImageProcessingTwo(CPFloatImage& F, float* r_in, float* b_in, float* g_in, 
						float* r_out, float* b_out, float* g_out)
{
	parallel_for(blocked_range2d<int, int>(0, F.h, 0, F.w),
		[&](const blocked_range2d<int>& r)
	{
		int baseX   = r.cols().begin();
		int baseY   = r.rows().begin();
		int numRows = r.rows().end() - baseY;
		int rowSpan = r.cols().end() - baseX;

		float *iRedBase   = r_in + (baseY * F.w) + baseX;
		float *iGreenBase = g_in + (baseY * F.w) + baseX;
		float *iBlueBase  = b_in + (baseY * F.w) + baseX;
		float *oRedBase   = r_out + (baseY * F.w) + baseX;
		float *oGreenBase = g_out + (baseY * F.w) + baseX;
		float *oBlueBase  = b_out + (baseY * F.w) + baseX;

		for (auto i = 0; i < numRows; i++,
			iRedBase += F.w, iGreenBase += F.w, iBlueBase += F.w,
			oRedBase += F.w, oGreenBase += F.w, oBlueBase += F.w)
		{
			float *iRedPtr   = iRedBase;
			float *iGreenPtr = iGreenBase;
			float *iBluePtr  = iBlueBase;
			float *oRedPtr   = oRedBase;
			float *oGreenPtr = oGreenBase;
			float *oBluePtr  = oBlueBase;

			for (auto j = 0; j < rowSpan; j++,
				iRedPtr++, iGreenPtr++, iBluePtr++,
				oRedPtr++, oGreenPtr++, oBluePtr++)
			{
				*oRedPtr = *iRedPtr / (*iRedPtr + *iGreenPtr + *iBluePtr);
				*oGreenPtr = *iGreenPtr / (*iRedPtr + *iGreenPtr + *iBluePtr);
				*oBluePtr = *iGreenPtr / 2;
			}
		}
	}
	);
}

void ImageProcessingThree(CPFloatImage& F, float* r_in, float* b_in, float* g_in,
						  float* r_out, float* b_out, float* g_out)
{
	parallel_for(blocked_range2d<int, int>(0, F.h, 0, F.w),
		[&](const blocked_range2d<int>& r)
	{
		int baseX   = r.cols().begin();
		int baseY   = r.rows().begin();
		int numRows = r.rows().end() - baseY;
		int rowSpan = r.cols().end() - baseX;

		float *iRedBase   = r_in + (baseY * F.w) + baseX;
		float *iGreenBase = g_in + (baseY * F.w) + baseX;
		float *iBlueBase  = b_in + (baseY * F.w) + baseX;
		float *oRedBase   = r_out + (baseY * F.w) + baseX;
		float *oGreenBase = g_out + (baseY * F.w) + baseX;
		float *oBlueBase  = b_out + (baseY * F.w) + baseX;

		for (auto i = 0; i < numRows; i++,
			iRedBase += F.w, iGreenBase += F.w, iBlueBase += F.w,
			oRedBase += F.w, oGreenBase += F.w, oBlueBase += F.w)
		{
			float *iRedPtr   = iRedBase;
			float *iGreenPtr = iGreenBase;
			float *iBluePtr  = iBlueBase;
			float *oRedPtr   = oRedBase;
			float *oGreenPtr = oGreenBase;
			float *oBluePtr  = oBlueBase;

			for (auto j = 0; j < rowSpan; j++,
				iRedPtr++, iGreenPtr++, iBluePtr++,
				oRedPtr++, oGreenPtr++, oBluePtr++)
			{
				float X = *iRedPtr * (*iBluePtr / *iGreenPtr);
				float Y = *iBluePtr;
				float Z = (1 - *iRedPtr - *iGreenPtr) * (*iBluePtr / *iGreenPtr);

				*oRedPtr = 3.2405f * X + -1.5371f * Y + -0.4985f * Z;
				*oGreenPtr = -0.9693f * X + 1.8760f * Y + 0.0416f * Z;
				*oBluePtr = 0.0556f * X + -0.2040f * Y + 1.0572f * Z;
			}
		}
	}
	);
}

int main(void)
{
	// Initialise COM so we can export image data using WIC
	HRESULT hr = initCOM();

	// COM failed!
	if (!SUCCEEDED(hr)) return 1;

	try
	{
		CPFloatImage F;
		loadImage(std::wstring(L"Images\\Barcelona_highres.jpg"), &F);

		float* redIn   = F.redChannel;
		float* greenIn = F.greenChannel;
		float* blueIn  = F.blueChannel;

		if (!redIn, !greenIn, !blueIn)
			throw std::exception("Input image not loaded!");

		std::cout << "Image loaded: width = " << F.w << ", height = " << F.h << std::endl;

		float* redOut	   = static_cast<float*>(malloc(F.w * F.h * sizeof(float)));
		float* greenOut    = static_cast<float*>(malloc(F.w * F.h * sizeof(float)));
		float* blueOut     = static_cast<float*>(malloc(F.w * F.h * sizeof(float)));

		float* redOutOne   = static_cast<float*>(malloc(F.w * F.h * sizeof(float)));
		float* greenOutOne = static_cast<float*>(malloc(F.w * F.h * sizeof(float)));
		float* blueOutOne  = static_cast<float*>(malloc(F.w * F.h * sizeof(float)));

		if (!redOut, !greenOut, !blueOut, !redOutOne, !greenOutOne, !blueOutOne)
			throw std::exception("Cannot create output image!");

		tick_count pt0 = tick_count::now();
		ImageProcessing(F, redIn, blueIn, greenIn, redOut, blueOut, greenOut);
		ImageProcessingTwo(F, redOut, blueOut, greenOut, redOutOne, blueOutOne, greenOutOne);
		ImageProcessingThree(F, redOutOne, blueOutOne, greenOutOne, redOut, blueOut, greenOut);
		tick_count pt1 = tick_count::now();

		delete redOutOne, blueOutOne, greenOutOne;

		std::cout << "parallel conversion operation took " << (pt1 - pt0).seconds() << " seconds"
				  << "\nSaving image...\n";

		saveImage(F.w, F.h, redOut, greenOut, blueOut, std::wstring(L"test_loadsave.tiff"));

		std::cout << "...done\n\n";
		shutdownCOM();
		return 0;
	}
	catch (std::exception& err)
	{
		std::cout << err.what() << std::endl;
		shutdownCOM();
		return 1;
	}
}