#ifndef MULTI_PASS_IMAGE_VIEW_H
#define MULTI_PASS_IMAGE_VIEW_H

#include <iostream>

#include <nanogui/common.h>
#include <nanogui/imageview.h>
#include <nanogui/screen.h>
#include <nanogui/window.h>

#include "ChannelEditsManager.h"
#include "ChannelSelector.h"
#include "GLTexture.h"
#include "JWH_Util.h"
#include "MaskEdit.h"

using namespace std;
using namespace Eigen;
using nanogui::ImageView;
using nanogui::Screen;
using nanogui::Window;
using nanogui::MatrixXu;
using nanogui::GLUniformBuffer;
using nanogui::Cursor;

class MultiPassImageView : public nanogui::ImageView {
    friend class EditTransferScreen;

   public:
    MultiPassImageView(Widget *parent,
                       shared_ptr<ChannelEditsManager> channelEditsManager,
                       shared_ptr<ChannelSelector> channelSelector,
                       vector<bool> *selectedChannels);

    ~MultiPassImageView();
    void initTexture(ImageDataTypeVec textures);
    void initPolygonMask(ImageDataType polyTexture);
    void initShader();
    void draw(NVGcontext *ctx) override;
    void writeFrameBuffer(nanogui::GLFramebuffer &fBuff, String filename);
    bool keyboardCharacterEvent(unsigned int codepoint) override;
    bool keyboardEvent(int key, int /*scancode*/, int action,
                       int modifiers) override;
    bool mouseButtonEvent(const Vector2i &p, int button, bool down,
                          int modifiers) override;
    bool mouseDragEvent(const Vector2i &p, const Vector2i &rel, int button,
                        int modifiers) override;
    bool scrollEvent(const Vector2i &p, const Vector2f &rel) override;
    bool mouseMotionEvent(const Vector2i &p, const Vector2i &rel, int button,
                          int modifiers) override;
    void drawSelection(NVGcontext *ctx);
    void drawBrush(NVGcontext *ctx);
    void performLayout(NVGcontext *ctx) override;

    int getSelectedChannelId();
    vector<int> getSelectedChannelIds();
    inline void setSelectedChannelId(int id)
    {
        mCurrentSelectionId = id;
        mCurrentSelectionOrderIndex = -1;
    }
    bool validMaskSelected();
    void clearCurrentSelection();

    void showMask();
    void writeParameters();

    inline bool rectSelectionValid() { return mRectSelection.valid(); };
    inline bool channelSelected()
    {
        return (getSelectedChannelIds()).size() > 0;
    };
    inline JWH::Rectangle getRect() { return mRectSelection; }
    inline int getSelectedMaskId() { return mCurrentMaskIndex; }
    inline void setSelectedMaskId(int id) { mCurrentMaskIndex = id; };
    inline void writeTempEditImages()
    {
        mSaveData = true;
        mWriteEditToTemp = true;
    };

    void setBrushState(bool state);
    void setBrushSize(int val) { mBushSize = val; }
    void setBrushValue(float val) { mBrushValue = val; }
    
    void drawMask(const Vector2i &p);
    Mat getPolygonMask() { return polygonMask; };



   private:
    // Variables
    ImageDataTypeVec mTextures;
    ImageDataType mPolyTexture;
    shared_ptr<ChannelEditsManager> mChannelEditsManager;
    shared_ptr<ChannelSelector> mChannelSelector;
    vector<bool> *mSelectedChannels;
    bool mToggleRGBAndPasses;  // false is RGB
    bool mToggleShowMask;      // true is mask
    bool mShowSelectedChannel;
    bool mLeftMouseDown;
    bool mRectVisible;
    bool mMaskActive;
    bool mSaveData;
    bool mInitFrameBuffer;
    bool mAKeyDown;
    bool mWriteEditToTemp;
    bool mBrushActive;
    bool mShowPolygon;
    GLuint mFrameBuffer;
    GLuint mEditTex;
    GLuint mEditDeltaTex;
    GLuint mMaskTex;
    GLuint mBeautyPassTex;
    Vector2i startMousePoint;
    Mat polygonMask;
    Vector2i mCursorPos;
    int mBushSize;
    float mBrushValue;

    Mat mEditBeautyPassMat;
    JWH::Rectangle mRectSelection;
    vector<int> mSelectionOrdering;
    int mCurrentSelectionId;  // -1 means use selection index from ordering
                              // below. This is set using mask selection tool
    int mCurrentSelectionOrderIndex;  // -1 means show composite only
    int mCurrentMaskIndex;            // -1 mean no mask is selected yet

    // Functions
    void selectChannel(Mat *polygon = nullptr);
    // void createMask();
    Vector4f getScreenSpaceRect();
    void saveImages();
};

#endif
