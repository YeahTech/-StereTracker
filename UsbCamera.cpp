#include "stdafx.h"
#include <assert.h>
#include "UsbCamera.h"

#ifndef USE_YUV422_FORMAT
#define USE_YUV422_FORMAT 0
#endif

#ifndef USE_RGB24_FORMAT
#define USE_RGB24_FORMAT  1
#endif

#ifndef SEND_WORK_STATE 
#define SEND_WORK_STATE  0
#endif

UsbCamera::UsbCamera():bisValid(false),pBuffer(NULL),pBYTEbuffer(NULL),bufferSize(0),bytePP(2.0), 
	connected(false),bnotify(false),width(0),height(0)
{

	m_camera = NULL;

	if (FAILED(CoInitialize(NULL))) 
	{
		return;
	}
	pGraph     = NULL;
	pDeviceFilter   = NULL;
	pMediaControl   = NULL;
	pSampleGrabberFilter = NULL;
	pSampleGrabber   = NULL;
	pGrabberInput   = NULL;
	pGrabberOutput   = NULL;
	pCameraOutput   = NULL;
	pMediaEvent    = NULL;
	pNullFilter    = NULL;
	pNullInputPin   = NULL;
	pBuilder    = NULL;

}

UsbCamera::~UsbCamera()
{
	if( connected ) 
	{
		if (pMediaControl )
		{
			pMediaControl->Stop();
		}
		SAFE_RELEASE(pGraph);
		SAFE_RELEASE(pDeviceFilter);
		SAFE_RELEASE(pMediaControl);
		SAFE_RELEASE(pSampleGrabberFilter);
		SAFE_RELEASE(pSampleGrabber);
		SAFE_RELEASE(pGrabberInput);
		SAFE_RELEASE(pGrabberOutput);
		SAFE_RELEASE(pCameraOutput);
		SAFE_RELEASE(pMediaEvent);
		SAFE_RELEASE(pNullFilter);
		SAFE_RELEASE(pNullInputPin);
		SAFE_RELEASE(pBuilder);
		CoUninitialize();
	}
	if( pBuffer )
		delete[] pBuffer;
	if( pBYTEbuffer )
		delete[] pBYTEbuffer;
}

void UsbCamera::Init(int deviceId, bool displayProperties,
	int framerate, int iw , int ih, string mode )
{
	HRESULT hr = S_OK;
	format_mode = mode;
	// Create the Filter Graph Manager.
	hr = CoCreateInstance(CLSID_FilterGraph, NULL, CLSCTX_INPROC,
		IID_IGraphBuilder, (void **)&pGraph);

	hr = CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER, 
		IID_IBaseFilter, (LPVOID *)&pSampleGrabberFilter);

	hr = pGraph->QueryInterface(IID_IMediaControl, (void **) &pMediaControl);
	hr = pGraph->QueryInterface(IID_IMediaEvent, (void **) &pMediaEvent);

	hr = CoCreateInstance(CLSID_NullRenderer, NULL, CLSCTX_INPROC_SERVER,
		IID_IBaseFilter, (LPVOID*) &pNullFilter);

	hr = pGraph->AddFilter(pNullFilter, L"NullRenderer");

	hr = pSampleGrabberFilter->QueryInterface(IID_ISampleGrabber, (void**)&pSampleGrabber);

	AM_MEDIA_TYPE   mt;
	ZeroMemory(&mt, sizeof(AM_MEDIA_TYPE));
	mt.majortype  = MEDIATYPE_Video;

	if (mode == "MEDIASUBTYPE_RGB24" )
	{
		mt.subtype = MEDIASUBTYPE_RGB24;
		bytePP     = 3.0;
	}
	else if (mode == "MEDIASUBTYPE_YUY2" )
	{
		mt.subtype = MEDIASUBTYPE_YUY2;
		bytePP     = 2.0;
	}

	mt.formattype = FORMAT_VideoInfo; 
	hr = pSampleGrabber->SetMediaType(&mt);

	pGraph->AddFilter(pSampleGrabberFilter, L"Grabber");

	// Bind Device Filter. We know the device because the id was passed in
	if(!BindFilter(deviceId, &pDeviceFilter)) 
	{
		ErrMsg(TEXT("未找到USB摄像头!\n请检查设备后重试!"));
		exit(0);
		return;
	}

	pGraph->AddFilter(pDeviceFilter, NULL);

	CComPtr<IEnumPins> pEnum;
	pDeviceFilter->EnumPins(&pEnum);
	hr = pEnum->Reset();
	hr = pEnum->Next(1, &pCameraOutput, NULL); 
	pEnum = NULL; 
	pSampleGrabberFilter->EnumPins(&pEnum);
	pEnum->Reset();
	hr = pEnum->Next(1, &pGrabberInput, NULL); 
	pEnum = NULL;
	pSampleGrabberFilter->EnumPins(&pEnum);
	pEnum->Reset();
	pEnum->Skip(1);
	hr = pEnum->Next(1, &pGrabberOutput, NULL); 
	pEnum = NULL;
	pNullFilter->EnumPins(&pEnum);
	pEnum->Reset();
	hr = pEnum->Next(1, &pNullInputPin, NULL);

	SetCrossBar(framerate,iw,ih,mode);

	if (displayProperties) 
	{
		CComPtr<ISpecifyPropertyPages> pPages;

		HRESULT hr = pCameraOutput->QueryInterface(IID_ISpecifyPropertyPages, (void**)&pPages);
		if (SUCCEEDED(hr))
		{
			PIN_INFO PinInfo;
			pCameraOutput->QueryPinInfo(&PinInfo);
			CAUUID caGUID;
			pPages->GetPages(&caGUID);
			OleCreatePropertyFrame( NULL,0, 0,L"Property Sheet",1,
				(IUnknown **)&(pCameraOutput),
				caGUID.cElems, caGUID.pElems,
				0,0,NULL );
			CoTaskMemFree(caGUID.pElems);
			PinInfo.pFilter->Release();
		}
	}

	hr = pGraph->Connect(pCameraOutput, pGrabberInput);
	hr = pGraph->Connect(pGrabberOutput, pNullInputPin);

	pSampleGrabber->SetBufferSamples(TRUE); // true for wait frame done call back 不再另外开辟单帧缓冲区
	pSampleGrabber->SetOneShot(TRUE); // FALSE=截图后继续运行graph,TRUE=STOP RUN GRAPH

	hr = pSampleGrabber->GetConnectedMediaType( &mt );
	VIDEOINFOHEADER *videoHeader;
	assert(mt.formattype == FORMAT_VideoInfo);
	videoHeader = reinterpret_cast<VIDEOINFOHEADER*>(mt.pbFormat);
	width  = videoHeader->bmiHeader.biWidth;
	height = videoHeader->bmiHeader.biHeight;
	FreeMediaType(mt);

	pMediaControl->Run();
	connected = true;
}

//将设备id为deviceId的设备 绑定到指定的pFilter
bool UsbCamera::BindFilter(int deviceId, IBaseFilter **pFilter)
{
	if (deviceId < 0)
		return false;
	CComPtr<ICreateDevEnum> pCreateDevEnum;
	HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER,
		IID_ICreateDevEnum, (void**)&pCreateDevEnum);
	if (hr != NOERROR)
		return false;

	CComPtr<IEnumMoniker> pEm;
	hr = pCreateDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory,
		&pEm, 0);
	if (hr != NOERROR) 
		return false;

	pEm->Reset();
	ULONG cFetched;
	IMoniker *pM;
	int index = 0;
	while(hr = pEm->Next(1, &pM, &cFetched), hr==S_OK, index <= deviceId)
	{
		IPropertyBag *pBag;
		hr = pM->BindToStorage(0, 0, IID_IPropertyBag, (void **)&pBag);
		if(SUCCEEDED(hr)) 
		{
			VARIANT var;
			var.vt = VT_BSTR;
			hr = pBag->Read(L"FriendlyName", &var, NULL);
			if (hr == NOERROR) 
			{
				if (index == deviceId)
				{
					pM->BindToObject(0, 0, IID_IBaseFilter, (void**)pFilter);
				}
				SysFreeString(var.bstrVal);
			}
			pBag->Release();
		}
		pM->Release();
		index++;
	}
	return true;
}

//对于有多个输入和输出的设备，需选择一个通路，另外设置媒体类型;
void UsbCamera::SetCrossBar(int fr, int iiw, int iih,string mode)
{
	IAMCrossbar *pXBar1             = NULL;
	IAMStreamConfig*      pVSC      = NULL;

	HRESULT hr = CoCreateInstance(CLSID_CaptureGraphBuilder2, NULL,
		CLSCTX_INPROC_SERVER, IID_ICaptureGraphBuilder2, 
		(void **)&pBuilder);

	if (SUCCEEDED(hr))
		hr = pBuilder->SetFiltergraph(pGraph);

	hr = pBuilder->FindInterface(&LOOK_UPSTREAM_ONLY, NULL, 
		pDeviceFilter,IID_IAMCrossbar, (void**)&pXBar1);

	if (SUCCEEDED(hr)) 
	{
		long OutputPinCount;
		long InputPinCount;
		long PinIndexRelated;
		long PhysicalType;
		long inPort = 0;
		long outPort = 0;

		pXBar1->get_PinCounts(&OutputPinCount,&InputPinCount);
		for(int i =0;i<InputPinCount;i++)
		{
			pXBar1->get_CrossbarPinInfo(TRUE,i,&PinIndexRelated,&PhysicalType);
			if(PhysConn_Video_Composite==PhysicalType) 
			{
				inPort = i;
				break;
			}
		}
		for(int i =0;i<OutputPinCount;i++)
		{
			pXBar1->get_CrossbarPinInfo(FALSE,i,&PinIndexRelated,&PhysicalType);
			if(PhysConn_Video_VideoDecoder==PhysicalType) 
			{
				outPort = i;
				break;
			}
		}

		if(S_OK==pXBar1->CanRoute(outPort,inPort))
		{
			pXBar1->Route(outPort,inPort);
		}
		pXBar1->Release();  
	}

	//设置媒体类型;
	hr = pBuilder->FindInterface( &PIN_CATEGORY_CAPTURE,0,pDeviceFilter,
		IID_IAMStreamConfig,(void**)&pVSC );

	AM_MEDIA_TYPE *pmt;
	if( SUCCEEDED(hr) )
	{
		hr = pVSC->GetFormat(&pmt);

		if (hr == NOERROR)
		{
			if (pmt->formattype == FORMAT_VideoInfo )
			{
				VIDEOINFOHEADER *pvi = (VIDEOINFOHEADER*) pmt->pbFormat;
				if (mode == "MEDIASUBTYPE_RGB24" )
					pmt->subtype = MEDIASUBTYPE_RGB24;
				else if (mode == "MEDIASUBTYPE_YUY2" )
					pmt->subtype = MEDIASUBTYPE_YUY2;

				pvi->AvgTimePerFrame = (LONGLONG)( 10000000 / fr );
				pvi->bmiHeader.biWidth  = iiw;
				pvi->bmiHeader.biHeight = iih;
				pvi->bmiHeader.biSizeImage = DIBSIZE(pvi->bmiHeader);
				hr = pVSC->SetFormat(pmt);
			} 
			FreeMediaType(*pmt);
		}
		SAFE_RELEASE( pVSC );
	}
}

HRESULT UsbCamera::GrabByteFrame()
{
	HRESULT hr;
	long    size = 0;
	long evCode;

	//hr = pMediaEvent->WaitForCompletion(10e4, &evCode); // INFINITE
	hr = pMediaEvent->WaitForCompletion(INFINITE, &evCode);


#if SEND_WORK_STATE
	if( evCode == EC_COMPLETE )
		pMediaControl->Pause();
	else if( FAILED(hr) || evCode <= 0 )
	{
		if( !bnotify ) 
		{
			bnotify = true;
			bisValid = false;
			return E_FAIL;
		}
	}
#endif

	pSampleGrabber->GetCurrentBuffer(&size, NULL);

	// use YUV422 format
#if USE_YUV422_FORMAT
	// if buffer is not the same size as before, create a new one
	if( size != bufferSize )
	{
		if( pBuffer )
			delete[] pBuffer;
		bufferSize = size;
		pBuffer = new long[bufferSize];
		if( pBYTEbuffer )
			delete[] pBYTEbuffer;

		pBYTEbuffer = new BYTE[bufferSize*2];
		memset(pBYTEbuffer,0,sizeof(BYTE)*bufferSize*2);
	}

	pSampleGrabber->GetCurrentBuffer(&size, pBuffer);

	const BYTE* pSrc = (BYTE*) pBuffer;
	const BYTE* pSrcEnd = pSrc + (width*height*2);
	BYTE* pDest = pBYTEbuffer;

	while (pSrc < pSrcEnd)
	{
		for (register int i =0; i< width; i++)
		{
			BYTE temp = *(pSrc++);
			BYTE temp2 = *(pSrc++);
			*(pDest++) = temp2;
			*(pDest++) = temp;
		}

	}  
#endif

#if USE_RGB24_FORMAT
	// use RGB format 
	if (size != bufferSize)
	{
		if (pBuffer)
			delete[] pBuffer;
		bufferSize = size;
		pBuffer = new long[bufferSize];
		if(pBYTEbuffer)
			delete[] pBYTEbuffer;
		pBYTEbuffer = new BYTE[bufferSize*3];
	}

	pSampleGrabber->GetCurrentBuffer(&size, pBuffer);

	BYTE* pDest = pBYTEbuffer;
	BYTE *pBYTETemp = pBYTEbuffer;
	const BYTE* pSrc = (BYTE*) pBuffer;


	const ULONG remainder = ((width*3+3) & ~3) - width*3;

	for (register unsigned int i = 0; i < height; i++ )
	{
		pDest = pBYTETemp + (height-i) * width * 3;
		for (register unsigned int j = 0; j < width; j++ )
		{
			const BYTE blue = *(pSrc++);
			const BYTE green = *(pSrc++);
			const BYTE red = *(pSrc++);

			*(pDest++) = red;
			*(pDest++) = green;
			*(pDest++) = blue;
			pDest += remainder;
		}
	} 
#endif

	return S_OK;
}

ImageBuffer UsbCamera::getImage()
{
	HRESULT hr = S_OK;
	hr = GrabByteFrame();

	if(FAILED(hr))
		ErrMsg(TEXT("UsbCamera disconnect!"));

	const BYTE* pSrc = GetByteBuffer();

#if USE_YUV422_FORMAT
	m_buffer = ImageBuffer( width,height,ImageBuffer::FORMAT_YUV422,
		(unsigned char*)pSrc,(int)(width*height*2.));
#endif

#if USE_RGB24_FORMAT
	m_buffer = ImageBuffer( width,height,ImageBuffer::FORMAT_RGB,
		(unsigned char*)pSrc,(int)(width*height*3.));
#endif

	pMediaControl->Run();

	return m_buffer;
}

UsbCamera* UsbCamera::getCamera(int port, int framerate /* = 30 */, int width /* = 320 */, 
	int height /* = 240  */,string mode)
{
	if (m_camera == NULL)
	{
		m_camera = new UsbCamera();
		m_camera->Init(port,false,framerate,width,height,mode); 
		m_camera->bisValid = true;
	}
	return m_camera;
}

void UsbCamera::destroyUsbCamera()
{
	if (m_camera)
	{
		delete m_camera;
		m_camera = NULL; 
	}
}

void UsbCamera::FreeMediaType(AM_MEDIA_TYPE& mt)
{
	if (mt.cbFormat != 0) 
	{
		CoTaskMemFree((PVOID)mt.pbFormat);
		mt.cbFormat = 0;
		mt.pbFormat = NULL;
	}
	if (mt.pUnk != NULL) 
	{
		mt.pUnk->Release();
		mt.pUnk = NULL;
	}
}

int UsbCamera::getWidth() const
{
	return width;
}

int UsbCamera::getHeight() const
{
	return height;
}

void UsbCamera::ErrMsg(LPTSTR szFormat,...)
{
	static TCHAR szBuffer[2048]={0};
	const size_t NUMCHARS = sizeof(szBuffer) / sizeof(szBuffer[0]);
	const int LASTCHAR = NUMCHARS - 1;
	va_list pArgs;
	va_start(pArgs, szFormat);
	_vsntprintf(szBuffer, NUMCHARS - 1, szFormat, pArgs);
	va_end(pArgs);

	szBuffer[LASTCHAR] = TEXT('\0');
	///MessageBox(NULL, szBuffer, "UsbCamera Error", 
	// MB_OK | MB_ICONEXCLAMATION | MB_TASKMODAL);
}

void UsbCamera::DisplayFilterProperties()
{
	ISpecifyPropertyPages* pProp;

	HRESULT hr = pDeviceFilter->QueryInterface(IID_ISpecifyPropertyPages, (void**)&pProp);
	if (SUCCEEDED(hr))
	{
		FILTER_INFO FilterInfo;
		hr = pDeviceFilter->QueryFilterInfo(&FilterInfo);
		IUnknown* pFilterUnk;
		pDeviceFilter->QueryInterface(IID_IUnknown,(void**)&pFilterUnk);

		CAUUID caGUID;
		pProp->GetPages(&caGUID);
		pProp->Release();
		OleCreatePropertyFrame(
			NULL,
			0,
			0,
			FilterInfo.achName,
			1,
			&pFilterUnk,
			caGUID.cElems,
			caGUID.pElems,
			0,
			0,
			NULL);
		pFilterUnk->Release();
		FilterInfo.pGraph->Release();
		CoTaskMemFree(caGUID.pElems);
	}

}