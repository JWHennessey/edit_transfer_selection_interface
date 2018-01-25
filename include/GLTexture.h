#ifndef GL_TEXTURE_H
#define GL_TEXTURE_H

#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>

#include <nanogui/glutil.h>
#include <nanogui/opengl.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#define GLDataType GL_FLOAT
using namespace cv;
using namespace std;
using PixelDataType = float;
const static GLint glInternalFormat = GL_RGB32F;
const static GLint glFormat = GL_BGR;
const static cv::ImreadModes ReadImageType = cv::IMREAD_UNCHANGED;
const static int NO_EDIT_PARAMS = 60;
const static int MAX_MASK_PER_CHANNEL = 10;

class MaskEdit;  // forward decare, included in cpp file

class GLTexture {
   public:
    using handleType = std::unique_ptr<PixelDataType[]>;

    GLTexture() = default;
    GLTexture(const std::string &textureName)
        : mTextureName(textureName), mTextureId(0)
    {
    }

    GLTexture(const std::string &textureName, GLint textureId)
        : mTextureName(textureName), mTextureId(textureId)
    {
    }

    GLTexture(const GLTexture &other) = delete;
    GLTexture(GLTexture &&other) noexcept
        : mTextureName(std::move(other.mTextureName))
        , mTextureId(other.mTextureId)
    {
        other.mTextureId = 0;
    }

    GLTexture &operator=(const GLTexture &other) = delete;
    GLTexture &operator=(GLTexture &&other) noexcept
    {
        mTextureName = std::move(other.mTextureName);
        std::swap(mTextureId, other.mTextureId);
        return *this;
    }
    ~GLTexture() noexcept
    {
        if (mTextureId) glDeleteTextures(1, &mTextureId);
    }

    GLuint texture() const { return mTextureId; }
    void setTextureId(int textureId) { mTextureId = textureId; }
    const std::string &textureName() const { return mTextureName; }

    /**
  *  Load a file in memory and create an OpenGL texture.
  *  Returns a handle type (an std::unique_ptr) to the loaded pixels.
  */
    virtual handleType load(const Mat &image);


    virtual void update(const Mat &image);

    // virtual handleType load(std::vector<MaskEdit> &edits);  // Correct load
    // for
    // 2D tex of edit
    // params
    GLuint mTextureId;

   private:
    std::string mTextureName;
};

class GLTexture2DArray : public GLTexture {
   public:
    GLTexture2DArray(const std::string &textureName)
        : GLTexture(textureName), mNoChannels(0)
    {
    }
    handleType load(const Mat &image) override { assert(false); };
    GLTexture::handleType init(int no_channels);
    void updateTexture(const int channel_id,
                       std::vector<shared_ptr<MaskEdit>> &edits,
                       PixelDataType *textureData);

   private:
    int mNoChannels;
};

class GL3DTexture : public GLTexture {
   public:
    GL3DTexture(const std::string &textureName) : GLTexture(textureName) {}
    handleType load(const Mat &image) override { assert(false); };
    handleType load(const std::vector<Mat> &images);  // Correct load for 3d tex
    handleType load(std::vector<shared_ptr<MaskEdit>>
                        &images);  // Correct load for 3d tex of masks
    void update(std::vector<shared_ptr<MaskEdit>> &images, bool all_masks);
};

void copyDataTo3DTexture(PixelDataType *textureData, const cv::Mat &image,
                         int start_index, int single_image_data_size);

using ImageDataType = pair<GLTexture, GLTexture::handleType>;
using ImageDataTypeVec = vector<pair<GLTexture, GLTexture::handleType>>;
using ImageData2DArray = pair<GLTexture2DArray, GLTexture::handleType>;
using Image3DDataType = pair<GL3DTexture, GLTexture::handleType>;
using Image3DDataTypeVec = vector<Image3DDataType>;

#endif
