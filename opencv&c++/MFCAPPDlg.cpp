#include "stdafx.h"
#include "MFCAPP.h"
#include "MFCAPPDlg.h"
#include "afxdialogex.h"
#include "conio.h"
#include <math.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <vector>  
#include <cstdio>  
#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace cv;
using namespace std;


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMFCAPPDlg 对话框



CMFCAPPDlg::CMFCAPPDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_MFCAPP_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMFCAPPDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CMFCAPPDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CMFCAPPDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CMFCAPPDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CMFCAPPDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CMFCAPPDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CMFCAPPDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CMFCAPPDlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &CMFCAPPDlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CMFCAPPDlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &CMFCAPPDlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON10, &CMFCAPPDlg::OnBnClickedButton10)
	ON_BN_CLICKED(IDC_BUTTON11, &CMFCAPPDlg::OnBnClickedButton11)
	ON_BN_CLICKED(IDC_BUTTON12, &CMFCAPPDlg::OnBnClickedButton12)
	ON_BN_CLICKED(IDC_BUTTON13, &CMFCAPPDlg::OnBnClickedButton13)
	ON_BN_CLICKED(IDC_BUTTON14, &CMFCAPPDlg::OnBnClickedButton14)
	ON_BN_CLICKED(IDC_BUTTON15, &CMFCAPPDlg::OnBnClickedButton15)
	ON_BN_CLICKED(IDC_BUTTON15, &CMFCAPPDlg::OnBnClickedButton15)
	ON_BN_CLICKED(IDC_BUTTON16, &CMFCAPPDlg::OnBnClickedButton16)
	ON_BN_CLICKED(IDC_BUTTON17, &CMFCAPPDlg::OnBnClickedButton17)
	ON_BN_CLICKED(IDC_BUTTON18, &CMFCAPPDlg::OnBnClickedButton18)
	ON_BN_CLICKED(IDC_BUTTON19, &CMFCAPPDlg::OnBnClickedButton19)
	ON_BN_CLICKED(IDC_BUTTON20, &CMFCAPPDlg::OnBnClickedButton20)
	ON_BN_CLICKED(IDC_BUTTON21, &CMFCAPPDlg::OnBnClickedButton21)
	ON_BN_CLICKED(IDC_BUTTON22, &CMFCAPPDlg::OnBnClickedButton22)
	ON_BN_CLICKED(IDC_BUTTON23, &CMFCAPPDlg::OnBnClickedButton23)
	ON_BN_CLICKED(IDC_BUTTON24, &CMFCAPPDlg::OnBnClickedButton24)
	ON_BN_CLICKED(IDC_BUTTON25, &CMFCAPPDlg::OnBnClickedButton25)
	ON_BN_CLICKED(IDC_BUTTON26, &CMFCAPPDlg::OnBnClickedButton26)
	ON_BN_CLICKED(IDC_BUTTON27, &CMFCAPPDlg::OnBnClickedButton27)
	ON_BN_CLICKED(IDC_BUTTON28, &CMFCAPPDlg::OnBnClickedButton28)
END_MESSAGE_MAP()


// CMFCAPPDlg 消息处理程序

BOOL CMFCAPPDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	AllocConsole();

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}



void CMFCAPPDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CMFCAPPDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CMFCAPPDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

Mat Conv(Mat inputs, float(*kernel)[3],  int kernel_size)
{
	Mat output;
	output.create(inputs.rows, inputs.cols, CV_8UC1);
	int idx = ceil(kernel_size / 2);
	for (int i = 1; i < inputs.rows - 1; i++)
	{
		for (int j = 1; j < inputs.cols - 1; j++)
		{
			output.at<uchar>(i, j) = 0;
			float val = 0.0;

			for (int m = 0; m < kernel_size; m++)
			{
				for (int n = 0; n < kernel_size; n++)
				{
					float k = kernel[m][n];
					int inp = inputs.at<uchar>(i + m - idx, j + n - idx);
					 val = val + inp*k ;
					 
				}
			}
			output.at<uchar>(i, j) = abs(val);
		}
	}

	return output;
}

Mat addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;		//盐噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;		//椒噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	return dstImage;
}

double generateGaussianNoise(double mu, double sigma)
{
	//定义最小值
	double epsilon = numeric_limits<double>::min();
	double z0 = 0, z1 = 0;
	bool flag = false;
	flag = !flag;
	if (!flag)
		return z1*sigma + mu;
	double u1, u2;
	do
	{
		u1 = rand()*(1.0 / RAND_MAX);
		u2 = rand()*(1.0 / RAND_MAX);
	} while (u1 <= epsilon);
	z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
	z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
	return z0*sigma + mu;
}

Mat addGaussianNoise(Mat& src)
{
	Mat result = src.clone();
	int channels = result.channels();
	int nRows = result.rows;
	int nCols = result.cols*channels;
	if (result.isContinuous())
	{
		nCols = nCols*nRows;
		nRows = 1;
	}
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			int val = result.ptr<uchar>(i)[j] + generateGaussianNoise(2, 0.8) * 32;
			if (val < 0)
				val = 0;
			if (val > 255)
				val = 255;
			result.ptr<uchar>(i)[j] = (uchar)val;
		}
	}
	return result;
}






void CMFCAPPDlg::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("E:/Git/opencv/山林.jpeg");
	imshow("raw_img", img);
}


void CMFCAPPDlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("E:/Git/opencv/山林.jpeg");
	Mat gray;
	//vector<int> hist(256);
	int hist[256];
	gray.create(img.rows, img.cols, CV_8UC1);
	for (int k = 0; k < 256; k++)
	{
		hist[k] = 0;
	}
	for (int i = 0 ; i < img.rows ; i++)
		{
		for (int j = 0 ; j < img.cols ; j++)
			{
			gray.at<uchar>(i, j) = 0.3*img.at<Vec3b>(i, j)[2] + 0.59*img.at<Vec3b>(i, j)[1] + 0.11*img.at<Vec3b>(i, j)[0];
			hist[gray.at<uchar>(i, j)] += 1.0;
			}
		}

	for (int m = 0; m < 256; m++)
		{
		_cprintf("hist[%d] = %d\n",m,hist[m]);
		//_cprintf("%d\n", hist[k]);
		}
	imshow("gray_img", gray);

}






void CMFCAPPDlg::OnBnClickedButton3()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("E:/Git/opencv/山林.jpeg");
	Mat gray;
	Mat equimg;
	//vector<int> hist(256);
	int hist[256],s,sum=0,su[256],val;
	gray.create(img.rows, img.cols, CV_8UC1);
	equimg.create(img.rows, img.cols, CV_8UC1);
	for (int k = 0; k < 256; k++)
	{
		hist[k] = 0;
	}
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			gray.at<uchar>(i, j) = 0.3*img.at<Vec3b>(i, j)[2] + 0.59*img.at<Vec3b>(i, j)[1] + 0.11*img.at<Vec3b>(i, j)[0];
			hist[gray.at<uchar>(i, j)] += 1.0;
		}
	}
	for (int i=0; i < 256; i++)
	{
		sum += hist[i];
		su[i] = sum;
	}

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			val = gray.at<uchar>(i, j);
			s = floor(255.0 * su[val] / (img.rows*img.cols));
			equimg.at<uchar>(i, j) = s;
		}
	}
	imshow("raw_gray", gray);
	imshow("equ_image", equimg);
}


void CMFCAPPDlg::OnBnClickedButton4()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("E:/Git/opencv/山林.jpeg");
	Mat gray;
	Mat sharpimg,laplsing;

	int val;
	gray.create(img.rows, img.cols, CV_8UC1);
	sharpimg.create(img.rows, img.cols, CV_8UC1);

	cvtColor(img, gray, CV_RGB2GRAY);

	for (int i = 0; i < img.rows-1; i++)
	{
		for (int j = 0; j < img.cols-1; j++)
		{
			sharpimg.at<uchar>(i, j) = saturate_cast<uchar>(abs(gray.at<uchar>(i, j) - gray.at<uchar>(i + 1, j) ) + abs(gray.at<uchar>(i, j) - gray.at<uchar>(i, j + 1)));
		}
	}

	imshow("raw_gray", gray);
	imshow("grad_image", sharpimg);
}


void CMFCAPPDlg::OnBnClickedButton5()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat imgraw = imread("E:/Git/lena.jfif");
	Mat img,edgecan;
	cvtColor(imgraw, img, CV_RGB2GRAY);
	edgecan.create(img.rows, img.cols, CV_8UC1);
	int robertx[2][2] = { {1,0}, {0,-1} };
	int roberty[2][2] = { { 0,1 }, { -1,0 } };
	float prewitt[3][3] = { { -1,0,1 } ,{ -1,0,1 } ,{ -1,0,1 } };
	float laplace[3][3] = { { 0,1,0 } ,{ 1,-4,1 } ,{ 0,1,0 } };
	float sobel[3][3] = { { -1,0,1 } ,{ -2,0,2 } ,{ -1,0,1 } };
 
	Mat edgepr = Conv(img, prewitt, 3);
	Mat edgela = Conv(img, laplace, 3);
	Mat edgeso = Conv(img, sobel, 3);
	Canny(img, edgecan, 50, 100, 3, false);

	imshow("lena_raw", img);
	imshow("prewitt", edgepr);
	imshow("laplace", edgela);
	imshow("sobel", edgeso);
	imshow("Canny", edgecan);

}



void CMFCAPPDlg::OnBnClickedButton6()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat imgraw = imread("E:/Git/lena.jfif");
	Mat saltimg = addSaltNoise(imgraw, 3000);
	imshow("raw_img", imgraw);
	imshow("AddSaltNoise", saltimg);
}


void CMFCAPPDlg::OnBnClickedButton7()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat imgraw = imread("E:/Git/lena.jfif");
	Mat img;
	cvtColor(imgraw, img, CV_RGB2GRAY);
	Mat saltimg = addSaltNoise(img, 3000);

	//Mat fliterav;
	//blur(saltimg, fliterav, Size(7, 7));
	float avg[3][3] = { { 1.0 / 9.0,1.0 / 9.0,1.0 / 9.0 } ,{ 1.0 / 9.0,1.0 / 9.0,1.0 / 9.0 } ,{ 1.0 / 9.0,1.0 / 9.0,1.0 / 9.0 } };
	Mat fliterav = Conv(img, avg, 3);
	imshow("avgfliter", fliterav);
}


void CMFCAPPDlg::OnBnClickedButton8()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat imgraw = imread("E:/Git/lena.jfif");
	Mat img;
	cvtColor(imgraw, img, CV_RGB2GRAY);
	Mat saltimg = addSaltNoise(img, 3000);
	Mat flitergau;
	GaussianBlur(img, flitergau, Size(5, 5), 0, 0);
	imshow("GaussianBlur", flitergau);
}


void CMFCAPPDlg::OnBnClickedButton9()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat imgraw = imread("E:/Git/lena.jfif");
	Mat guasimg = addGaussianNoise(imgraw);
	imshow("raw_img", imgraw);
	imshow("AddGuassNoise", guasimg);
}


void CMFCAPPDlg::OnBnClickedButton10()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat imgraw = imread("E:/Git/lena.jfif");
	Mat img;
	cvtColor(imgraw, img, CV_RGB2GRAY);
	Mat w1,w2,w3,w4,w5,w6,w7,w8,out;
	out.create(img.rows, img.cols, CV_8UC1);
	float m1[3][3] = { { 1.0 / 4.0,1.0 / 4.0,0.0 } ,{ 1.0 / 4.0,1.0 / 4.0,0.0 } ,{ 0.0,0.0,0.0 } };
	float m2[3][3] = { { 0.0,0.0,0.0 }, { 1.0 / 4.0,1.0 / 4.0,0.0 } ,{ 1.0 / 4.0,1.0 / 4.0,0.0 } };
	float m3[3][3] = { { 0.0,1.0 / 4.0,1.0 / 4.0 } ,{ 0.0,1.0 / 4.0,1.0 / 4.0 } ,{ 0.0,0.0,0.0 } };
	float m4[3][3] = { { 0.0,0.0,0.0 } ,{ 0.0,1.0 / 4.0,1.0 / 4.0 } ,{ 0.0,1.0 / 4.0,1.0 / 4.0 } };
	float m5[3][3] = { { 1.0 / 6.0,1.0 / 6.0,1.0 / 6.0 } ,{ 1.0 / 6.0,1.0 / 6.0,1.0 / 6.0 } ,{ 0.0,0.0,0.0 } };
	float m6[3][3] = { { 0.0,0.0,0.0 } ,{ 1.0 / 6.0,1.0 / 6.0,1.0 / 6.0 } ,{ 1.0 / 6.0,1.0 / 6.0,1.0 / 6.0 } };
	float m7[3][3] = { { 1.0 / 6.0,1.0 / 6.0,0.0 } ,{ 1.0 / 6.0,1.0 / 6.0,0.0 } ,{ 1.0 / 6.0,1.0 / 6.0,0.0 } };
	float m8[3][3] = { { 0.0,1.0 / 6.0,1.0 / 6.0 } ,{ 0.0,1.0 / 6.0,1.0 / 6.0 } ,{ 0.0,1.0 / 6.0,1.0 / 6.0 } };
	w1 = Conv(img, m1, 3);
	w2 = Conv(img, m2, 3);
	w3 = Conv(img, m3, 3);
	w4 = Conv(img, m4, 3);
	w5 = Conv(img, m5, 3);
	w6 = Conv(img, m6, 3);
	w7 = Conv(img, m7, 3);
	w8 = Conv(img, m8, 3);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0;j<img.cols;j++)
		{ 
			int val = min({ abs(img.at<uchar>(i, j) -w1.at<uchar>(i, j)),
				abs(img.at<uchar>(i, j) -w2.at<uchar>(i, j)),
				abs(img.at<uchar>(i, j) -w3.at<uchar>(i, j)),
				abs(img.at<uchar>(i, j) -w4.at<uchar>(i, j)),
				abs(img.at<uchar>(i, j) -w5.at<uchar>(i, j)),
				abs(img.at<uchar>(i, j) -w6.at<uchar>(i, j)),
				abs(img.at<uchar>(i, j) -w7.at<uchar>(i, j)),
				abs(img.at<uchar>(i, j) -w8.at<uchar>(i, j)) });
			val = abs(img.at<uchar>(i, j) - val);
			out.at<uchar>(i, j) = val;
		}
	}
	imshow("raw_img", imgraw);
	imshow("Side window filter", out);

}


void CMFCAPPDlg::OnBnClickedButton11()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("E:/Git/open.jpg");
	Mat structure_element = getStructuringElement(0, Size(13, 13));
	Mat dst_erode, dst_dilate;
	dst_erode.create(img.rows, img.cols, CV_8UC1);
	dst_dilate.create(img.rows, img.cols, CV_8UC1);

	erode(img, dst_erode, structure_element);
	dilate(dst_erode, dst_dilate, structure_element);

	imshow("src", img);
	imshow("dst_erode", dst_erode);
	imshow("dst_dilate", dst_dilate);
	waitKey(0);
}


void CMFCAPPDlg::OnBnClickedButton12()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat imgraw = imread("E:/Git/lena.jfif");
	Mat img;
	cvtColor(imgraw, img, CV_RGB2GRAY);
	Mat saltimg = addSaltNoise(img, 3000);
	Mat flitermed;
	flitermed.create(img.rows, img.cols, CV_8UC1);
	medianBlur(saltimg, flitermed, 7);
	imshow("AddSaltNoise", saltimg);
	imshow("Medi Fliter", flitermed);

}


void CMFCAPPDlg::OnBnClickedButton13()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat src = imread("E:/Git/lena.jfif");
	Point2f srcTri[3], dstTri[3]; //二维坐标下的点，类型为浮点
	Mat rot_matr(2, 3, CV_32FC1); //单通道矩阵
	Mat warp_mat(2, 3, CV_32FC1);
	Mat dst1;
	dst1 = Mat::zeros(src.rows, src.cols, src.type());
	//计算矩阵仿射变换
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src.cols - 1, 0); //缩小一个像素
	srcTri[2] = Point2f(0, src.rows - 1);
	//改变目标图像大小
	dstTri[0] = Point2f(src.cols * 0.0, src.rows * 0.33);
	dstTri[1] = Point2f(src.cols * 0.85, src.rows * 0.25);
	dstTri[2] = Point2f(src.cols* 0.15, src.rows* 0.7);
	//由三对点计算仿射变换
	warp_mat = getAffineTransform(srcTri, dstTri);
	//对图像做仿射变换
	warpAffine(src, dst1, warp_mat, src.size());
	imshow("src", src);
	imshow("dst_rot1", dst1);


	Mat dst;
	dst = Mat::zeros(src.rows, src.cols, src.type());
	Point2f center = Point2f(src.cols / 2, src.rows / 2);
	double angle = -50.0; //旋转角度，负值表示顺时针
	double scale = 0.6; //各项同性的尺度因子
	Mat rot_mat = getRotationMatrix2D(center, angle, scale);
	warpAffine(src, dst, rot_mat,src.size());//将src仿射变换存入dst
	imshow("src", src);
	imshow("dst_rot2", dst);

}


void CMFCAPPDlg::OnBnClickedButton14()
{
	// TODO: 在此添加控件通知处理程序代码
	Point2f srcQuad[4], dstQuad[4];
	Mat src = imread("E:/Git/carline.jpg");
	Mat warp_matrix(3, 3, CV_32FC1);
	Mat dst;
	dst = Mat::zeros(src.rows, src.cols, src.type());
	srcQuad[0] = Point2f(0, 0); //src top left
	srcQuad[1] = Point2f(src.cols - 1, 0); //src top right
	srcQuad[2] = Point2f(0, src.rows - 1); //src bottom left
	srcQuad[3] = Point2f(src.cols - 1, src.rows - 1); //src bot right
	dstQuad[0] = Point2f(src.cols*0.05, src.rows*0.33); //dst top left
	dstQuad[1] = Point2f(src.cols*0.9, src.rows*0.25); //dst top right
	dstQuad[2] = Point2f(src.cols*0.2, src.rows*0.7); //dst bottom left
	dstQuad[3] = Point2f(src.cols*0.8, src.rows*0.9); //dst bot right
	warp_matrix = getPerspectiveTransform(srcQuad, dstQuad);
	warpPerspective(src, dst, warp_matrix, src.size());
	imshow("src", src);
	imshow("dst_per", dst);
}




void CMFCAPPDlg::OnBnClickedButton15()
{
	// TODO: 在此添加控件通知处理程序代码
	// TODO: 在此添加控件通知处理程序代码
	ifstream fin("E://Git//MFCAPP//imageDatalist_right.txt"); /* 标定所用图像文件的路径 */
	ofstream fout("E://Git//MFCAPP//caliberation_result_right.txt");  /* 保存标定结果的文件 */
																					//读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化	
	_cprintf("start to find corners………………\n");
	int image_count = 0;  /* 图像数量 */
	Size image_size;  /* 图像的尺寸 */
	Size board_size = Size(9, 6);    /* 标定板上每行、列的角点数 */
	vector<Point2f> image_points_buf;  /* 缓存每幅图像上检测到的角点 */
	vector<vector<Point2f>> image_points_seq; /* 保存检测到的所有角点 */
	string filename;
	int count = -1;//用于存储角点个数。
	while (getline(fin, filename))
	{
		image_count++;
		// 用于观察检验输出
		_cprintf("image_count = %d\n", image_count);

		Mat imageInput = imread(filename);
		if (image_count == 1)  //读入第一张图片时获取图像宽高信息
		{
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			_cprintf("image_size.width = %d\n", image_size.width);
			_cprintf("image_size.height = %d\n", image_size.height);
		}

		/* 提取角点 */
		if (0 == findChessboardCorners(imageInput, board_size, image_points_buf))
		{
			_cprintf("can not find chessboard corners!\n"); //找不到角点
			waitKey(0);
			exit(1);
		}
		else
		{
			Mat view_gray;
			cvtColor(imageInput, view_gray, CV_RGB2GRAY);
			/* 亚像素精确化 */
			find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5)); //对粗提取的角点进行精确化
																			//cornerSubPix(view_gray,image_points_buf,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,30,0.1));
			image_points_seq.push_back(image_points_buf);  //保存亚像素角点
														   /* 在图像上显示角点位置 */
			drawChessboardCorners(view_gray, board_size, image_points_buf, false); //用于在图片中标记角点
			imshow("Camera Calibration", view_gray);//显示图片
			waitKey(50);//暂停0.5S		
		}
	}
	int total = image_points_seq.size();
	_cprintf("total = %d\n", total);
	int CornerNum = board_size.width*board_size.height;  //每张图片上总的角点数
	for (int ii = 0; ii<total; ii++)
	{
		if (0 == ii%CornerNum)// 24 是每幅图片的角点个数。此判断语句是为了输出 图片号，便于控制台观看 
		{
			int i = -1;
			i = ii / CornerNum;
			int j = i + 1;
			_cprintf("--> the %d image's data --> :", j);
		}
		if (0 == ii % 3)	// 此判断语句，格式化输出，便于控制台查看
		{
			_cprintf("\n");
		}
		else
		{
			cout.width(10);
		}
		//输出所有的角点
		_cprintf(" -->%f", image_points_seq[ii][0].x);
		_cprintf(" -->%f", image_points_seq[ii][0].y);
	}
	_cprintf("find corner successfully！\n");

	//以下是摄像机标定
	_cprintf("start calibration………………");
	/*棋盘三维信息*/
	Size square_size = Size(10, 10);  /* 实际测量得到的标定板上每个棋盘格的大小 */
	vector<vector<Point3f>> object_points; /* 保存标定板上角点的三维坐标 */
										   /*内外参数*/
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 摄像机内参数矩阵 */
	vector<int> point_counts;  // 每幅图像中角点的数量
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;  /* 每幅图像的旋转向量 */
	vector<Mat> rvecsMat; /* 每幅图像的平移向量 */
						  /* 初始化标定板上角点的三维坐标 */
	int i, j, t;
	for (t = 0; t<image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i<board_size.height; i++)
		{
			for (j = 0; j<board_size.width; j++)
			{
				Point3f realPoint;
				/* 假设标定板放在世界坐标系中z=0的平面上 */
				realPoint.x = i*square_size.width;
				realPoint.y = j*square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
	for (i = 0; i<image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
	/* 开始标定 */
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
	_cprintf("calibrate successfully！\n");
	//对标定结果进行评价
	_cprintf("start evaluate the calibration results………………\n");
	double total_err = 0.0; /* 所有图像的平均误差的总和 */
	double err = 0.0; /* 每幅图像的平均误差 */
	vector<Point2f> image_points2; /* 保存重新计算得到的投影点 */
	_cprintf("\t each image's calibration error：\n");
	_cprintf("each image's calibration error：\n");
	for (i = 0; i<image_count; i++)
	{
		vector<Point3f> tempPointSet = object_points[i];
		/* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f> tempImagePoint = image_points_seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		_cprintf("the %d th image's average err: %f pixels\n", i + 1, err);
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	_cprintf(" the total average err：%f \n", total_err / image_count);
	fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;
	_cprintf("evaluate successfully！\n");
	//保存定标结果  	
	_cprintf("start save calibration results………………\n");
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
	fout << "相机内参数矩阵：" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "畸变系数：\n";
	fout << distCoeffs << endl << endl << endl;
	for (int i = 0; i<image_count; i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << tvecsMat[i] << endl;
		/* 将旋转向量转换为相对应的旋转矩阵 */
		Rodrigues(tvecsMat[i], rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << rvecsMat[i] << endl << endl;
	}
	_cprintf("save successfully\n");
	fout << endl;
	/************************************************************************
	显示定标结果
	*************************************************************************/
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	_cprintf("save rectified image\n");
	string imageFileName;
	std::stringstream StrStm;
	for (int i = 0; i != image_count; i++)
	{
		_cprintf("Frame # %d ...\n", i + 1);
		initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
		StrStm.clear();
		imageFileName.clear();
		string filePath = "E://Git//MFCAPP//chess";
		StrStm << i + 1;
		StrStm >> imageFileName;
		filePath += imageFileName;
		filePath += ".jpg";
		Mat imageSource = imread("E://Git//MFCAPP//left01.jpg");
		Mat newimage = imageSource.clone();
		//另一种不需要转换矩阵的方式
		//undistort(imageSource,newimage,cameraMatrix,distCoeffs);
		remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);
		StrStm.clear();
		filePath.clear();
		filePath = "E://Git//MFCAPP//chess";
		StrStm << i + 1;
		StrStm >> imageFileName;
		filePath += imageFileName;
		filePath += "_d.jpg";
		imwrite(filePath, newimage);
	}
	_cprintf(" save successfully\n");
	waitKey(0);
}

void thresplit(Mat img,int thre)
{

	Mat gray;
	Mat sharpimg, laplsing;

	int val;
	gray.create(img.rows, img.cols, CV_8UC1);
	sharpimg.create(img.rows, img.cols, CV_8UC1);

	cvtColor(img, gray, CV_RGB2GRAY);

	for (int i = 0; i < img.rows - 1; i++)
	{
		for (int j = 0; j < img.cols - 1; j++)
		{
			if (gray.at<uchar>(i, j) > thre)
			{
				sharpimg.at<uchar>(i, j) = gray.at<uchar>(i, j);
			}
			else
			{
				sharpimg.at<uchar>(i, j) = 0;
			}
		}
	}

	imshow("raw_gray", gray);
	imshow("grad_image", sharpimg);
}



void CMFCAPPDlg::OnBnClickedButton16()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("E:/Git/lena.jfif");
	int thre = 120;
	thresplit(img,thre);
}

int getmaxT()
{
	int hist[256];
	//******************
	Mat img = imread("E:/Git/lena.jfif");
	Mat gray;
	gray.create(img.rows, img.cols, CV_8UC1);
	for (int k = 0; k < 256; k++)
	{
		hist[k] = 0;
	}
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			gray.at<uchar>(i, j) = 0.3*img.at<Vec3b>(i, j)[2] + 0.59*img.at<Vec3b>(i, j)[1] + 0.11*img.at<Vec3b>(i, j)[0];
			hist[gray.at<uchar>(i, j)] += 1.0;
		}
	}
	float u0, u1, w0, w1;
	int count0, t, maxT;
	float devi, maxDevi = 0; //方差及最大方差
	int i, sum = 0;
	//****************
	for (i = 0; i < 256; i++) 
	{
		sum = sum + hist[i];
	}
	for (t = 0; t < 255; t++) 
	{
		u0 = 0; count0 = 0;
		for (i = 0; i <= t; i++) //值为t时， cO组的均值及产生的
		{
			u0 += i*hist[i]; count0 += hist[i];
		}
		u0 = u0 / count0; w0 = (float)count0 / sum;
		for (i = t + 1; i < 256; i++) //刀阔值为t 时， cl组的均值及产生
		{
			u1+= i * hist[i];
		}
		u1 = u1 / (sum - count0); w1 = 1 - w0;
		devi = w0*w1 * (u1 - u0) * (u1 - u0); //两类间方差
		if (devi > maxDevi) //记录最大的方
		{
			maxDevi = devi;
			maxT = t;
		}
	}
	return maxT;
}
void CMFCAPPDlg::OnBnClickedButton17()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("E:/Git/lena.jfif");
	int maxT = getmaxT();
	thresplit(img,maxT);

}

int kittle(Mat img)
{
	Mat gray;
	gray.create(img.rows, img.cols, CV_8UC1);
	cvtColor(img, gray, CV_RGB2GRAY);

	int grads = 0;
	int sumgrads = 0;
	int sumgraygrads = 0;

	for (int i = 1; i < gray.rows - 1; i++)
	{
		for (int j = 1; j < gray.cols - 1; j++)
		{
			grads = max(fabs(gray.at<uchar>(i + 1, j) - gray.at<uchar>(i - 1, j)), fabs(gray.at<uchar>(i, j + 1) - gray.at<uchar>(i, j - 1)));
			sumgrads = sumgrads + grads;
			sumgraygrads = sumgraygrads + grads*(gray.at<uchar>(i, j));
		}
	}
	int KT = sumgraygrads / sumgrads;
	return KT;
}

void CMFCAPPDlg::OnBnClickedButton18()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("E:/Git/lena.jfif");
	int KT = kittle(img);
	thresplit(img,KT);
}


string convertToString(double d)
{
	ostringstream os;
	if (os << d)
		return os.str();
	return "invalid conversion";
}

Mat slicewnd(Mat img,int i, int j, int w, int h)
{
	Mat wnd;
	wnd.create(h, w, CV_8UC3);
	for (int k = 0; k < h; k++)
	{
		for (int m = 0; m < w; m++)
		{
			wnd.at<Vec3b>(k, m)[0] = img.at<Vec3b>(k + i, m + j)[0];
			wnd.at<Vec3b>(k, m)[1] = img.at<Vec3b>(k + i, m + j)[1];
			wnd.at<Vec3b>(k, m)[2] = img.at<Vec3b>(k + i, m + j)[2];
		}
	}
	return wnd;
}

void CMFCAPPDlg::OnBnClickedButton19()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat rgbimg,rgbimg1,hsvimg, hsvimg1,wnd,gray,gray1;
	MatND hist_img, hist_img1;
	rgbimg = imread("E:/Git/MFCAPP/giraffe.jpg");
	rgbimg1 = imread("E:/Git/MFCAPP/giraffe1.jpg");
	gray.create(rgbimg.rows, rgbimg.cols, CV_8UC1); 
	gray1.create(rgbimg1.rows, rgbimg1.cols, CV_8UC1);
	//cvtColor(rgbimg, gray, CV_RGB2GRAY);
	//cvtColor(rgbimg1, gray1, CV_RGB2GRAY);
	cvtColor(rgbimg, hsvimg, COLOR_RGB2HSV);
	cvtColor(rgbimg1, hsvimg1, COLOR_RGB2HSV);
	int w = hsvimg1.cols;
	int h = hsvimg1.rows;
	int step = 5;
	double score = 0.0;
	int trageti = 0;
	int tragetj = 0;
	//int h_bins = 361;
	int s_bins = 256;
	//int v_bins = 256;
	int histsize[] = { s_bins};
	//hue varies from 0 to 179,saturation from 0 to 255
	//float h_ranges[] = { 0,360 };
	float s_ranges[] = { 0,256 };
	//float v_ranges[] = { 0,256 };
	int dim = 1;
	const float* histRanges[] = { s_ranges};
	//use the 0-th and 1-st channels
	int channels[] = { 1};
	for (int i = 0; i < hsvimg.rows-h; i = i + step)
	{
		for (int j = 0; j < hsvimg.cols-w;j = j+ step)
		{
			wnd = slicewnd(hsvimg, i, j, w,h );
			calcHist(&wnd, 1, channels, Mat(), hist_img, dim, histsize, histRanges, true, false);
			calcHist(&hsvimg1, 1, channels, Mat(), hist_img1, dim, histsize, histRanges, true, false);

			normalize(hist_img, hist_img, 0, 1, NORM_MINMAX, -1, Mat());//归一化
			normalize(hist_img1, hist_img1, 0, 1, NORM_MINMAX, -1, Mat());

			double basetest1 = compareHist(hist_img, hist_img1, CV_COMP_BHATTACHARYYA);

			if (basetest1 >= score)
			{
				score = basetest1;
				trageti = i;
				tragetj = j;
			}
		}
	}

	rectangle(rgbimg, Point(trageti, tragetj), Point(trageti + h, tragetj + w), Scalar(0, 0, 255), 2);


	
	//double basebase = compareHist(hist_img, hist_img, CV_COMP_BHATTACHARYYA);


	//putText(rgbimg, convertToString(basebase), Point(50, 50), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, CV_AA);
	putText(rgbimg, convertToString(score), Point(50, 50), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, CV_AA);

	//namedWindow("img", CV_WINDOW_AUTOSIZE);
	//namedWindow("img1", CV_WINDOW_AUTOSIZE);

	imshow("hsvimg", rgbimg);
	imshow("hsvimg1", rgbimg1);
}


void CMFCAPPDlg::OnBnClickedButton20()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat obj = imread("E:/Git/MFCAPP/1.1.jpg");   //载入目标图像
	Mat scene = imread("E:/Git/MFCAPP/1.2.jpg"); //载入场景图像
	if (obj.empty() || scene.empty())
	{
		cout << "Can't open the picture!\n";
		return;
	}
	vector<KeyPoint> obj_keypoints, scene_keypoints;
	Mat obj_descriptors, scene_descriptors;
	Ptr<ORB> detector = ORB::create();

	detector->detect(obj, obj_keypoints);
	detector->detect(scene, scene_keypoints);
	detector->compute(obj, obj_keypoints, obj_descriptors);
	detector->compute(scene, scene_keypoints, scene_descriptors);

	BFMatcher matcher(NORM_HAMMING, true); //汉明距离做为相似度度量
	vector<DMatch> matches;
	matcher.match(obj_descriptors, scene_descriptors, matches);
	Mat match_img;
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img);
	imshow("滤除误匹配前", match_img);

	//保存匹配对序号
	vector<int> queryIdxs(matches.size()), trainIdxs(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		queryIdxs[i] = matches[i].queryIdx;
		trainIdxs[i] = matches[i].trainIdx;
	}

	Mat H12;   //变换矩阵

	vector<Point2f> points1;
	KeyPoint::convert(obj_keypoints, points1, queryIdxs);
	vector<Point2f> points2;
	KeyPoint::convert(scene_keypoints, points2, trainIdxs);
	int ransacReprojThreshold = 5;  //拒绝阈值


	H12 = findHomography(Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold);
	vector<char> matchesMask(matches.size(), 0);
	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);
	for (size_t i1 = 0; i1 < points1.size(); i1++)  //保存‘内点’
	{
		if (norm(points2[i1] - points1t.at<Point2f>((int)i1, 0)) <= ransacReprojThreshold) //给内点做标记
		{
			matchesMask[i1] = 1;
		}
	}
	Mat match_img2;   //滤除‘外点’后
	drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img2, Scalar(0, 0, 255), Scalar::all(-1), matchesMask);

	//画出目标位置
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0); obj_corners[1] = Point(obj.cols, 0);
	obj_corners[2] = Point(obj.cols, obj.rows); obj_corners[3] = Point(0, obj.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H12);
	//line( match_img2, scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
	//line( match_img2, scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
	//line( match_img2, scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
	//line( match_img2, scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
	line(match_img2, Point2f((scene_corners[0].x + static_cast<float>(obj.cols)), (scene_corners[0].y)), Point2f((scene_corners[1].x + static_cast<float>(obj.cols)), (scene_corners[1].y)), Scalar(0, 0, 255), 2);
	line(match_img2, Point2f((scene_corners[1].x + static_cast<float>(obj.cols)), (scene_corners[1].y)), Point2f((scene_corners[2].x + static_cast<float>(obj.cols)), (scene_corners[2].y)), Scalar(0, 0, 255), 2);
	line(match_img2, Point2f((scene_corners[2].x + static_cast<float>(obj.cols)), (scene_corners[2].y)), Point2f((scene_corners[3].x + static_cast<float>(obj.cols)), (scene_corners[3].y)), Scalar(0, 0, 255), 2);
	line(match_img2, Point2f((scene_corners[3].x + static_cast<float>(obj.cols)), (scene_corners[3].y)), Point2f((scene_corners[0].x + static_cast<float>(obj.cols)), (scene_corners[0].y)), Scalar(0, 0, 255), 2);

	float A_th;
	A_th = atan(abs((scene_corners[3].y - scene_corners[0].y) / (scene_corners[3].x - scene_corners[0].x)));
	A_th = 90 - 180 * A_th / 3.14;
	_cprintf("angle=%f\n", A_th);

	imshow("滤除误匹配后", match_img2);

	//line( scene, scene_corners[0],scene_corners[1],Scalar(0,0,255),2);
	//line( scene, scene_corners[1],scene_corners[2],Scalar(0,0,255),2);
	//line( scene, scene_corners[2],scene_corners[3],Scalar(0,0,255),2);
	//line( scene, scene_corners[3],scene_corners[0],Scalar(0,0,255),2);

	imshow("场景图像", scene);

	Mat rotimage;
	Mat rotate = getRotationMatrix2D(Point(scene.cols / 2, scene.rows / 2), A_th, 1);
	warpAffine(scene, rotimage, rotate, scene.size());
	imshow("rotimage", rotimage);


	//方法三 透视变换  
	Point2f src_point[4];
	Point2f dst_point[4];
	src_point[0].x = scene_corners[0].x;
	src_point[0].y = scene_corners[0].y;
	src_point[1].x = scene_corners[1].x;
	src_point[1].y = scene_corners[1].y;
	src_point[2].x = scene_corners[2].x;
	src_point[2].y = scene_corners[2].y;
	src_point[3].x = scene_corners[3].x;
	src_point[3].y = scene_corners[3].y;


	dst_point[0].x = 0;
	dst_point[0].y = 0;
	dst_point[1].x = obj.cols;
	dst_point[1].y = 0;
	dst_point[2].x = obj.cols;
	dst_point[2].y = obj.rows;
	dst_point[3].x = 0;
	dst_point[3].y = obj.rows;

	Mat newM(3, 3, CV_32FC1);
	newM = getPerspectiveTransform(src_point, dst_point);

	Mat dst = scene.clone();

	warpPerspective(scene, dst, newM, obj.size());

	imshow("obj", obj);
	imshow("dst", dst);

	Mat resultimg = dst.clone();

	absdiff(obj, dst, resultimg);//当前帧跟前面帧相减

	imshow("result", resultimg);

	imshow("dst", dst);
	imshow("src", obj);
}



void CMFCAPPDlg::OnBnClickedButton21()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat src1 = imread("E:/Git/MFCAPP/1.1.jpg", 1);
	Mat src2 = imread("E:/Git/MFCAPP/1.2.jpg", 1);
	imshow("src1", src1);
	imshow("src2", src2);

	if (!src1.data || !src2.data)
	{
		_cprintf(" --(!) Error reading images \n");
		return;
	}

	//sift feature detect  
	Ptr<SIFT> siftdetector = SIFT::create();
	vector<KeyPoint> kp1, kp2;

	siftdetector->detect(src1, kp1);
	siftdetector->detect(src2, kp2);
	Mat des1, des2;//descriptor  
	siftdetector->compute(src1, kp1, des1);
	siftdetector->compute(src2, kp2, des2);
	Mat res1, res2;

	drawKeypoints(src1, kp1, res1);//在内存中画出特征点  
	drawKeypoints(src2, kp2, res2);

	_cprintf("size of description of Img1: %d\n", kp1.size());
	_cprintf("size of description of Img2: %d\n", kp2.size());

	Mat transimg1, transimg2;
	transimg1 = res1.clone();
	transimg2 = res2.clone();

	char str1[20], str2[20];
	sprintf_s(str1, "%d", kp1.size());
	sprintf_s(str2, "%d", kp2.size());

	const char* str = str1;
	putText(transimg1, str1, Point(280, 230), 0, 1.0, Scalar(255, 0, 0), 2);//在图片中输出字符   

	str = str2;
	putText(transimg2, str2, Point(280, 230), 0, 1.0, Scalar(255, 0, 0), 2);//在图片中输出字符   

																			//imshow("Description 1",res1);  
	imshow("descriptor1", transimg1);
	imshow("descriptor2", transimg2);

	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches;
	matcher.match(des1, des2, matches);
	Mat img_match;
	drawMatches(src1, kp1, src2, kp2, matches, img_match);//,Scalar::all(-1),Scalar::all(-1),vector<char>(),drawmode);  
	_cprintf("number of matched points: %d\n", matches.size());
	imshow("matches", img_match);
	waitKey(10);
}

Mat g_srcImage, g_tempalteImage, g_resultImage;
int g_nMatchMethod;
int g_nMaxTrackbarNum = 5;

void on_matching(int, void*)
{
	Mat srcImage;
	g_srcImage.copyTo(srcImage);
	int resultImage_cols = g_srcImage.cols - g_tempalteImage.cols + 1;
	int resultImage_rows = g_srcImage.rows - g_tempalteImage.rows + 1;
	g_resultImage.create(resultImage_cols, resultImage_rows, CV_32FC1);

	matchTemplate(g_srcImage, g_tempalteImage, g_resultImage, g_nMatchMethod);
	normalize(g_resultImage, g_resultImage, 0, 1, NORM_MINMAX, -1, Mat());
	double minValue, maxValue;
	Point minLocation, maxLocation, matchLocation;
	minMaxLoc(g_resultImage, &minValue, &maxValue, &minLocation, &maxLocation);

	if (g_nMatchMethod == TM_SQDIFF || g_nMatchMethod == CV_TM_SQDIFF_NORMED)
	{
		matchLocation = minLocation;
	}
	else
	{
		matchLocation = maxLocation;
	}

	rectangle(srcImage, matchLocation, Point(matchLocation.x + g_tempalteImage.cols, matchLocation.y + g_tempalteImage.rows), Scalar(0, 0, 255), 2, 8, 0);
	rectangle(g_resultImage, matchLocation, Point(matchLocation.x + g_tempalteImage.cols, matchLocation.y + g_tempalteImage.rows), Scalar(0, 0, 255), 2, 8, 0);

	imshow("原始图", srcImage);
	imshow("效果图", g_resultImage);

}

void CMFCAPPDlg::OnBnClickedButton22()
{
	// TODO: 在此添加控件通知处理程序代码
	g_srcImage = imread("E:/Git/MFCAPP/giraffe.jpg");
	if (!g_srcImage.data)
	{
		cout << "原始图读取失败" << endl;

	}
	g_tempalteImage = imread("E:/Git/MFCAPP/giraffe1.jpg");
	if (!g_tempalteImage.data)
	{
		cout << "模板图读取失败" << endl;

	}

	imshow("g_srcImage", g_srcImage);
	imshow("g_tempalteImage", g_tempalteImage);

	namedWindow("原始图", CV_WINDOW_AUTOSIZE);
	namedWindow("效果图", CV_WINDOW_AUTOSIZE);
	createTrackbar("方法", "原始图", &g_nMatchMethod, g_nMaxTrackbarNum, on_matching);

	on_matching(0, NULL);


	waitKey(0);



}


void CMFCAPPDlg::OnBnClickedButton23()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat src = imread("E:/Git/lena.jfif", IMREAD_GRAYSCALE);
	namedWindow("input", CV_WINDOW_AUTOSIZE);
	imshow("input", src);
	Mat src_f;
	src.convertTo(src_f, CV_32F);
	// 参数初始化
	int kernel_size = 3;
	double sigma = 1.0, lambd = CV_PI / 8, gamma = 0.5, psi = 0;
	vector<Mat> destArray;
	double theta[4];
	Mat temp;
	// theta 法线方向
	theta[0] = 0;
	theta[1] = CV_PI / 4;
	theta[2] = CV_PI / 2;
	theta[3] = CV_PI - CV_PI / 4;
	// gabor 纹理检测器，可以更多，
	// filters = number of thetas * number of lambd
	// 这里lambad只取一个值，所有4个filter
	for (int i = 0; i < 4; i++)
	{
		Mat kernel1;
		Mat dest;
		kernel1 = getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta[i], lambd, gamma, psi, CV_32F);
		filter2D(src_f, dest, CV_32F, kernel1);
		destArray.push_back(dest);
	}
	// 显示与保存
	Mat dst1, dst2, dst3, dst4;
	convertScaleAbs(destArray[0], dst1);
	imwrite("E:/Git/MFCAPP/gabor1.jpg", dst1);
	convertScaleAbs(destArray[1], dst2);
	imwrite("E:/Git/MFCAPP/gabor2.jpg", dst2);
	convertScaleAbs(destArray[2], dst3);
	imwrite("E:/Git/MFCAPP/gabor3.jpg", dst3);
	convertScaleAbs(destArray[3], dst4);
	imwrite("E:/Git/MFCAPP/gabor4.jpg", dst4);
	imshow("gabor1", dst1);
	imshow("gabor2", dst2);
	imshow("gabor3", dst3);
	imshow("gabor4", dst4);
	// 合并结果
	add(destArray[0], destArray[1], destArray[0]);
	add(destArray[2], destArray[3], destArray[2]);
	add(destArray[0], destArray[2], destArray[0]);
	Mat dst;
	convertScaleAbs(destArray[0], dst, 0.2, 0);
	// 二值化显示
	Mat gray, binary;
	// cvtColor(dst, gray, COLOR_BGR2GRAY);
	threshold(dst, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	imshow("result", dst);
	imshow("binary", binary);
	imwrite("E:/Git/MFCAPP/result_01.png", binary);
	waitKey(0);
}

void elbp(Mat& src, Mat& dst, int radius, int neighbors)
{

	for (int n = 0; n < neighbors; n++)
	{
		// 采样点的计算
		float x = static_cast<float>(-radius * sin(2.0 * CV_PI * n / static_cast<float>(neighbors)));
		float y = static_cast<float>(radius * cos(2.0 * CV_PI * n / static_cast<float>(neighbors)));
		// 上取整和下取整的值
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// 小数部分
		float ty = y - fy;
		float tx = x - fx;
		// 设置插值权重
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) * ty;
		float w4 = tx * ty;
		// 循环处理图像数据
		for (int i = radius; i < src.rows - radius; i++)
		{
			for (int j = radius; j < src.cols - radius; j++)
			{
				// 计算插值
				float t = static_cast<float>(w1 * src.at<uchar>(i + fy, j + fx) + w2 * src.at<uchar>(i + fy, j + cx) + w3 * src.at<uchar>(i + cy, j + fx) + w4 * src.at<uchar>(i + cy, j + cx));
				// 进行编码
				dst.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) || (abs(t - src.at<uchar>(i, j)) < numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

void elbp1(Mat& src, Mat& dst)
{
	// 循环处理图像数据
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			uchar tt = 0;
			int tt1 = 0;
			uchar u = src.at<uchar>(i, j);
			if (src.at<uchar>(i - 1, j - 1) > u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i - 1, j) > u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i - 1, j + 1) > u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i, j + 1) > u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i + 1, j + 1) > u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i + 1, j) > u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i + 1, j - 1) > u) { tt += 1 << tt1; }
			tt1++;
			if (src.at<uchar>(i - 1, j) > u) { tt += 1 << tt1; }
			tt1++;

			dst.at<uchar>(i - 1, j - 1) = tt;
		}
	}
}

void CMFCAPPDlg::OnBnClickedButton24()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = cv::imread("E:/Git/lena.jfif", 0);
	namedWindow("image");
	imshow("image", img);

	int radius, neighbors;
	radius = 1;
	neighbors = 8;

	//创建一个LBP
	//注意为了溢出，我们行列都在原有图像上减去2个半径
	Mat dst = Mat(img.rows - 2 * radius, img.cols - 2 * radius, CV_8UC1, Scalar(0));
	elbp1(img, dst);
	namedWindow("normal");
	imshow("normal", dst);

	Mat dst1 = Mat(img.rows - 2 * radius, img.cols - 2 * radius, CV_8UC1, Scalar(0));
	elbp(img, dst1, 1, 8);
	namedWindow("circle");
	imshow("circle", dst1);

	waitKey(0);
}


void CMFCAPPDlg::OnBnClickedButton25()
{
	// TODO: 在此添加控件通知处理程序代码

	// TODO: 在此添加控件通知处理程序代码
	int iWidth = 512, iheight = 512;
	Mat matImg = Mat::zeros(iheight, iWidth, CV_8UC3);//三色通道
													  //1.获取样本
	int labels[5] = { 1.0, -1.0, -1.0, -1.0,1.0 }; //样本数据  
	Mat labelsMat(5, 1, CV_32SC1, labels);     //样本标签  
	float trainingData[5][2] = { { 501, 300 },{ 255, 10 },{ 501, 255 },{ 10, 501 },{ 450,500 } }; //Mat结构特征数据  
	Mat trainingDataMat(5, 2, CV_32FC1, trainingData);   //Mat结构标签  
														 //2.设置SVM参数
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);//可以处理非线性分割的问题
	svm->setKernel(ml::SVM::POLY);//径向基函数SVM::LINEAR
								  /*svm->setGamma(0.01);
								  svm->setC(10.0);*/
								  //算法终止条件
	svm->setDegree(1.0);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
	//3.训练支持向量
	svm->train(trainingDataMat, ml::SampleTypes::ROW_SAMPLE, labelsMat);
	//4.保存训练器
	svm->save("mnist_svm.xml");
	//5.导入训练器
	//Ptr<SVM> svm1 = StatModel::load<SVM>("mnist_dataset/mnist_svm.xml");

	//读取测试数据
	Mat sampleMat;
	Vec3b green(0, 255, 0), blue(255, 0, 0);
	for (int i = 0; i < matImg.rows; i++)
	{
		for (int j = 0; j < matImg.cols; j++)
		{
			sampleMat = (Mat_<float>(1, 2) << j, i);
			float fRespone = svm->predict(sampleMat);
			if (fRespone == 1)
			{
				matImg.at<cv::Vec3b>(i, j) = green;
			}
			else if (fRespone == -1)
			{
				matImg.at<cv::Vec3b>(i, j) = blue;
			}


		}
	}
	// Show the training data  
	int thickness = -1;
	int lineType = 8;
	for (int i = 0; i < trainingDataMat.rows; i++)
	{
		if (labels[i] == 1)
		{
			circle(matImg, Point(trainingData[i][0], trainingData[i][1]), 5, Scalar(0, 0, 0), thickness, lineType);
		}
		else
		{
			circle(matImg, Point(trainingData[i][0], trainingData[i][1]), 5, Scalar(255, 255, 255), thickness, lineType);
		}
	}

	//显示支持向量点
	thickness = 2;
	lineType = 8;
	Mat vec = svm->getSupportVectors();
	int nVarCount = svm->getVarCount();//支持向量的维数
	_cprintf("vec.rows=%d vec.cols=%d\n", vec.rows, vec.cols);
	for (int i = 0; i < vec.rows; ++i)
	{
		int x = (int)vec.at<float>(i, 0);
		int y = (int)vec.at<float>(i, 1);
		_cprintf("vec.at=%d %f,%f\n", i, vec.at<float>(i, 0), vec.at<float>(i, 1));
		_cprintf("x=%d,y=%d\n", x, y);
		circle(matImg, Point(x, y), 6, Scalar(0, 0, 255), thickness, lineType);
	}


	imshow("circle", matImg); // show it to the user  
	waitKey(0);
}


void CMFCAPPDlg::OnBnClickedButton26()
{
	// TODO: 在此添加控件通知处理程序代码
	// TODO: 在此添加控件通知处理程序代码
	//winsize(64,128),blocksize(16,16),blockstep(8,8),cellsize(8,8),bins9
	HOGDescriptor hog(Size(14, 14), Size(7, 7), Size(1, 1), Size(7, 7), 9);
	//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人
	Ptr<ml::SVM> svm = ml::SVM::create();//SVM分类器
	string ImgName;//图片名(绝对路径)
	ifstream fin("E:/Git/MFCAPP/sample/train_num.txt");//正样本图片的文件名列表
	if (!fin)
	{
		_cprintf("Pos/Neg imglist reading failed...\n");
		return;
	}
	for (int num = 0; num < 10 && getline(fin, ImgName); num++)
	{
		_cprintf("Now processing original image: %s\n", ImgName);
		Mat src = imread(ImgName);//读取图片
		if (src.empty())
			_cprintf("no pic\n");
		resize(src, src, Size(28, 28), 1);

		vector<float> descriptors;//HOG描述子向量
		hog.compute(src, descriptors, Size(5, 5));//计算HOG描述子，检测窗口移动步长(8,8)
												  //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
		if (num == 0)
		{
			DescriptorDim = descriptors.size();
			_cprintf("DescriptorDim: %d\n", DescriptorDim);
			//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
			sampleFeatureMat = Mat::zeros(10, DescriptorDim, CV_32FC1);
			//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
			sampleLabelMat = Mat::zeros(10, 1, CV_32SC1);//sampleLabelMat的数据类型必须为有符号整数型
		}
		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for (int i = 0; i < DescriptorDim; i++)
		{
			sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素
		}
		sampleLabelMat.at<int>(num, 0) = num;//正样本类别为1，有人
	}
	fin.close();
	//输出样本的HOG特征向量矩阵到文件
	svm->setType(ml::SVM::C_SVC);
	svm->setC(0.01);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6));
	_cprintf("Starting training...\n");
	svm->train(sampleFeatureMat, ml::ROW_SAMPLE, sampleLabelMat);//训练分类器
																 //将训练好的SVM模型保存为xml文件
	svm->SVM::save("E:/Git/MFCAPP/sample/SVM_HOG.xml");
	_cprintf("Finishing training...\n");

	/*
	Mat sampleMat;
	sampleMat.create(1, DescriptorDim, CV_32FC1);
	for (int i = 0; i < DescriptorDim; i++)
	{
	sampleMat.at<float>(0, i) = sampleFeatureMat.at<float>(5, i);
	}
	int ret = svm->predict(sampleMat);
	*/

	//imshow("src", src);
	waitKey();
}


void CMFCAPPDlg::OnBnClickedButton27()
{
	// TODO: 在此添加控件通知处理程序代码
	Ptr<ml::SVM> svm1 = ml::SVM::load("E:/Git/MFCAPP/sample/SVM_HOG.xml");

	if (svm1->empty())
	{
		_cprintf("load svm detector failed!!!\n");
		return;
	}

	Mat testimg;
	testimg = imread("E:/Git/MFCAPP/sample/9/0.png");
	resize(testimg, testimg, Size(28, 28), 1);
	imshow("src", testimg);
	//waitKey(0);

	HOGDescriptor hog(Size(14, 14), Size(7, 7), Size(1, 1), Size(7, 7), 9);
	vector<float> imgdescriptor;
	hog.compute(testimg, imgdescriptor, Size(5, 5));
	Mat sampleMat;
	sampleMat.create(1, imgdescriptor.size(), CV_32FC1);

	for (int i = 0; i < imgdescriptor.size(); i++)
	{
		sampleMat.at<float>(0, i) = imgdescriptor[i];//第num个样本的特征向量中的第i个元素
	}
	int ret = svm1->predict(sampleMat);
	_cprintf("ret=%d\n", ret);

}

double sumxy(Mat img,int x, int y)
{
	double sum = 0;
	for (int i =0;i<x;i++)
		for (int j = 0; j < y; j++)
		{
			sum = sum + img.at<uchar>(i, j);
		}
	return sum;
}

void CMFCAPPDlg::OnBnClickedButton28()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat rgbimg, gray,sum_img,haar_img;
	int h = 4, w = 2,step = 2;
	rgbimg = imread("E:/Git/lena.jfif");
	sum_img.create(rgbimg.rows, rgbimg.cols, CV_8UC1);
	haar_img.create(rgbimg.rows, rgbimg.cols, CV_8UC1);
	cvtColor(rgbimg, gray, COLOR_RGB2GRAY);
	for (int i = h;i<gray.rows-1;i = i+step)
		for (int j = w; j<gray.cols-1; j = j+step)
		{ 
			sum_img.at<uchar>(i, j) = (sumxy(gray, i, j) - sumxy(gray, i - h, j) - sumxy(gray, i, j - w) + sumxy(gray, i - h, j - w)) - (sumxy(gray, i, j + w) - sumxy(gray, i - h, j + w) - sumxy(gray, i, j - w + w) + sumxy(gray, i - h, j - w + w));
		}
	normalize(sum_img, haar_img, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
	imshow("raw",rgbimg);
	imshow("haar",haar_img);
}
