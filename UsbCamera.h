#ifndef USBCAMER_H_INCLUDE
#define USBCAMER_H_INCLUDE

#include <windows.h>
#include <dshow.h>
#include <atlbase.h>
#include <qedit.h>
#include <string>
//#include "ImageSource.h"



#define WIN32_LEAN_AND_MEAN

#ifndef SAFE_RELEASE
#define SAFE_RELEASE( x ) \
	if ( NULL != x ) \
{ \
	x->Release( ); \
	x = NULL; \
}
#endif

using namespace std;


class ImageBuffer
{
public:
	enum {
		FORMAT_YUV444=0,
		FORMAT_YUV422,
		FORMAT_YUV411,
		FORMAT_RGB,
		FORMAT_MONO,
		FORMAT_MONO16,
		FORMAT_UYV
	};    

	int width;              
	int height;              
	int format;             
	int size;                
	unsigned char* buffer; 


	ImageBuffer::ImageBuffer()
	{

	}
	ImageBuffer::ImageBuffer(int width, int height, int format, 
		unsigned char* buffer, int size)
		: width(width), height(height), format(format), buffer(buffer), size(size)
	{

	}
};


class ImageSource
{
public:
	virtual ImageBuffer getImage() =0;
	virtual int getWidth() const=0;
	virtual int getHeight() const=0;
};

class UsbCamera : public ImageSource
{
public:
	UsbCamera();
	virtual ~UsbCamera();

	virtual ImageBuffer getImage();
	virtual int   getWidth() const;
	virtual int   getHeight() const;

	 UsbCamera* getCamera(  int port = 0, int framerate = 30, int width = 320, 
		int height = 240, string mode = "" );
	 void   destroyUsbCamera();

	void    Init( int deviceId, bool displayProperties = false, int framerate = 30,int iw = 320, int ih = 240, string mode = "" );
	void    DisplayFilterProperties();

	bool    BindFilter(int deviceId, IBaseFilter **pFilter);
	void    SetCrossBar( int fr = 30, int iiw = 320, int iih = 240, string mode = "" );
	HRESULT    GrabByteFrame();
	long    GetBufferSize()  { return bufferSize; }
	long*    GetBuffer()   { return pBuffer;    }
	BYTE*    GetByteBuffer()  { return pBYTEbuffer;}

public:
	bool        bisValid;

protected:
	IGraphBuilder*   pGraph;
	IBaseFilter*   pDeviceFilter;
	IMediaControl*   pMediaControl;
	IBaseFilter*   pSampleGrabberFilter;
	ISampleGrabber*   pSampleGrabber;
	IPin*     pGrabberInput;
	IPin*     pGrabberOutput;
	IPin*     pCameraOutput;
	IMediaEvent*   pMediaEvent;
	IBaseFilter*   pNullFilter;
	IPin*     pNullInputPin;
	ICaptureGraphBuilder2*  pBuilder;

	UsbCamera*  m_camera;
	ImageBuffer    imagebuf;

	double                  bytePP;
private:
	void     ErrMsg(LPTSTR szFormat,...);
	void     FreeMediaType(AM_MEDIA_TYPE& mt);

	long  bufferSize;
	long*  pBuffer;
	BYTE*  pBYTEbuffer;
	bool  connected;
	int   width;
	int   height;

	ImageBuffer m_buffer;
	bool  bnotify;
	string     format_mode; 

};

#endif