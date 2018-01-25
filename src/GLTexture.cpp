
#include "GLTexture.h"
#include "MaskEdit.h"

/**
 *
 *  GLTexture
 *
 */

GLTexture::handleType GLTexture::load(const Mat &image)
{
    if (mTextureId) {
        glDeleteTextures(1, &mTextureId);
        mTextureId = 0;
    }
    int force_channels = 0;

    // Mat image = imread(fileName, ReadImageType);
    int w = image.cols;
    int h = image.rows;
    int n = image.channels();

    PixelDataType *image_data = (PixelDataType *)image.data;
    auto image_data_size =
        sizeof(PixelDataType) * (w * h * n);  // Byte size of image
    handleType textureData =
        make_unique<PixelDataType[]>(image_data_size);  // init unique pointer
    memcpy(textureData.get(), image_data, image_data_size);  // copy pixel data

    if (!textureData)
        throw std::invalid_argument("Could not load texture data from cv::mat");
    glGenTextures(1, &mTextureId);
    glBindTexture(GL_TEXTURE_2D, mTextureId);
    glTexImage2D(GL_TEXTURE_2D, 0, glInternalFormat, w, h, 0, glFormat,
                 GLDataType, textureData.get());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    return textureData;
}

void GLTexture::update(const Mat &image)
{
    int w = image.cols;
    int h = image.rows;
    int n = image.channels();

    glBindTexture(GL_TEXTURE_2D, mTextureId);

    glTexSubImage2D(GL_TEXTURE_2D,
                    0,            // Mipmap number
                    0, 0,         // xoffset, yoffset
                    w, h,         // width, height
                    glFormat,     // format
                    GLDataType,   // type
                    image.data);  // pointer to data

}

/**
 *
 *  GLTexture2DArray
 *
 */
GLTexture::handleType GLTexture2DArray::init(int no_channels)
{
    mNoChannels = no_channels;
    // cout << mTextureName << endl;
    if (mTextureId) {
        glDeleteTextures(1, &mTextureId);
        mTextureId = 0;
        // cout << "GLTexture::load id set to zero ";
    }

    Mat params_image(MAX_MASK_PER_CHANNEL, NO_EDIT_PARAMS, CV_32FC1, Scalar(0));
    // Mat image = imread(fileName, ReadImageType);
    int w = NO_EDIT_PARAMS;
    int h = MAX_MASK_PER_CHANNEL;

    auto param_image_size = w * h;
    auto param_image_byte_size = sizeof(PixelDataType) * param_image_size;
    auto params_vector_byte_size = param_image_byte_size * no_channels;

    handleType textureData = make_unique<PixelDataType[]>(
        params_vector_byte_size);  // init unique pointer

    for (auto i = 0; i < mNoChannels; i++) {
        auto start_index = param_image_size * i;
        PixelDataType *image_data = (PixelDataType *)params_image.data;

        memcpy(textureData.get() + start_index, image_data,
               param_image_byte_size);  // copy pixel data
    }
    if (!textureData)
        throw std::invalid_argument("Could not load texture data from cv::mat");
    glGenTextures(1, &mTextureId);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mTextureId);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F, NO_EDIT_PARAMS,
                 MAX_MASK_PER_CHANNEL, mNoChannels, 0, GL_RED, GLDataType,
                 textureData.get());

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    return textureData;
}

void GLTexture2DArray::updateTexture(const int channel_id,
                                     std::vector<shared_ptr<MaskEdit>> &edits,
                                     PixelDataType *textureData)
{

    if (mTextureId) {
        glDeleteTextures(1, &mTextureId);
        mTextureId = 0;
    }

    auto param_image_size = NO_EDIT_PARAMS * MAX_MASK_PER_CHANNEL;
    auto param_image_byte_size = sizeof(PixelDataType) * param_image_size;
    auto params_vector_byte_size = param_image_byte_size * mNoChannels;

    const Mat &params = edits[0]->getEditParamRef();

    int w = params.cols;
    int h = edits.size();

    auto single_row_data_size = sizeof(PixelDataType) * w;
    auto all_params_data_size = single_row_data_size * (h);

    PixelDataType *image_data = (PixelDataType *)params.data;

    auto start_index = param_image_size * channel_id;

    memcpy(textureData + start_index, image_data, single_row_data_size);

    for (auto i = 1; i < h; i++) {
        start_index += w;
        const Mat &params = edits[i]->getEditParamRef();
        PixelDataType *image_data = (PixelDataType *)params.data;
        memcpy(textureData + start_index, image_data, single_row_data_size);
    }

    if (!textureData)
        throw std::invalid_argument("Could not load texture data from cv::mat");
    glGenTextures(1, &mTextureId);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mTextureId);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_R32F, NO_EDIT_PARAMS,
                 MAX_MASK_PER_CHANNEL, mNoChannels, 0, GL_RED, GLDataType,
                 textureData);

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

/**
 * GL3DTexture
 */

GLTexture::handleType GL3DTexture::load(const std::vector<Mat> &images)
{
    // GLuint localTextureId = texture();
    if (!mTextureId) {
        glGenTextures(1, &mTextureId);
        /*        glDeleteTextures(1, &mTextureId);*/
        /*setTextureId(0);*/
    }
    int force_channels = 0;

    /*    Mat image = imread(fileNames[custom_index],*/
    /*ReadImageType);  // Read image to set size variable*/
    Mat image = images[0];
    /*    auto window_name = "Temp Window";*/
    // imshow(window_name, image);
    /*waitKey();*/

    int w = image.cols;
    int h = image.rows;
    int n = image.channels();

    int img_count = images.size();
    auto single_image_pixel_size = (w * h * n);
    auto single_image_data_size =
        sizeof(PixelDataType) * single_image_pixel_size;
    auto all_image_data_size = single_image_data_size * img_count;

    handleType textureData = make_unique<PixelDataType[]>(
        all_image_data_size);  // init unique pointer

    copyDataTo3DTexture(textureData.get(), image, 0,
                        single_image_data_size);  // zero index

    // Read all other images and copy data
    for (auto i = 1; i < img_count; i++) {
        Mat image = images[i];  // imread(fileNames[i], ReadImageType);
                                /*        imshow(window_name, image);*/
        /*waitKey();*/
        auto start_index = single_image_pixel_size * i;
        copyDataTo3DTexture(textureData.get(), image, start_index,
                            single_image_data_size);  // Sepecifc index
    }

    if (!textureData)
        throw std::invalid_argument(
            "Could not load texture data from filenames");

    glBindTexture(GL_TEXTURE_3D, mTextureId);

    glTexImage3D(GL_TEXTURE_3D, 0, glInternalFormat, w, h, img_count, 0,
                 glFormat, GLDataType, textureData.get());

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    return textureData;
}

GLTexture::handleType GL3DTexture::load(
    std::vector<shared_ptr<MaskEdit>> &images)
{
    // GLuint localTextureId = texture();
    if (mTextureId) {
        glDeleteTextures(1, &mTextureId);
        setTextureId(0);
    }
    int force_channels = 0;

    /*    Mat image = imread(fileNames[custom_index],*/
    /*ReadImageType);  // Read image to set size variable*/
    const Mat &image = images[0]->getMaskRef();
    /*    auto window_name = "Temp Window";*/
    // imshow(window_name, image);
    /*waitKey();*/

    // imshow("tex image", image);

    int w = image.cols;
    int h = image.rows;
    int n = image.channels();

    int img_count = images.size();
    auto single_image_pixel_size = (w * h * n);
    auto single_image_data_size =
        sizeof(PixelDataType) * single_image_pixel_size;
    auto all_image_data_size = single_image_data_size * img_count;

    handleType textureData = make_unique<PixelDataType[]>(
        all_image_data_size);  // init unique pointer

    copyDataTo3DTexture(textureData.get(), image, 0,
                        single_image_data_size);  // zero index

    // Read all other images and copy data
    for (auto i = 1; i < img_count; i++) {
        const Mat &image =
            images[i]->getMaskRef();  // imread(fileNames[i], ReadImageType);
                                      /*        imshow(window_name, image);*/
        /*waitKey();*/
        auto start_index = single_image_pixel_size * i;
        copyDataTo3DTexture(textureData.get(), image, start_index,
                            single_image_data_size);  // Sepecifc index
    }

    if (!textureData)
        throw std::invalid_argument(
            "Could not load texture data from filenames");

    glGenTextures(1, &mTextureId);
    glBindTexture(GL_TEXTURE_3D, mTextureId);

    glTexImage3D(GL_TEXTURE_3D, 0, glInternalFormat, w, h, img_count, 0,
                 glFormat, GLDataType, textureData.get());

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    return textureData;
}

void GL3DTexture::update(std::vector<shared_ptr<MaskEdit>> &images, bool all_masks)
{
    const Mat &image = images[0]->getMaskRef();

    int w = image.cols;
    int h = image.rows;
    int n = image.channels();
    int img_count = images.size();

    glBindTexture(GL_TEXTURE_3D, mTextureId);

    for (auto i = 0; i < img_count; i++) {
        const Mat &image =
            images[i]->getMaskRef();  // imread(fileNames[i], ReadImageType);

        if (!images[i]->getUpdateTexture() && !all_masks) continue;

        glTexSubImage3D(GL_TEXTURE_3D,
                        0,            // Mipmap number
                        0, 0, i,      // xoffset, yoffset, zoffset
                        w, h, 1,      // width, height, depth
                        glFormat,     // format
                        GLDataType,   // type
                        image.data);  // pointer to data
        images[i]->setUpdateTexture(false);
    }
}

void copyDataTo3DTexture(PixelDataType *textureData, const cv::Mat &image,
                         int start_index, int single_image_data_size)
{
    PixelDataType *image_data = (PixelDataType *)image.data;
    memcpy(textureData + start_index, image_data, single_image_data_size);
}
