
#include "MultiPassImageView.h"
#include "ChannelStat.h"
#include "EditTransferScreen.h"

MultiPassImageView::MultiPassImageView(
    Widget *parent, shared_ptr<ChannelEditsManager> channelEditsManager,
    shared_ptr<ChannelSelector> channelSelector, vector<bool> *selectedChannels)
    : nanogui::ImageView(parent, 0)
    , mChannelEditsManager(channelEditsManager)
    , mChannelSelector(channelSelector)
    , mSelectedChannels(selectedChannels)
    , mToggleRGBAndPasses(true)
    , mToggleShowMask(false)
    , mShowSelectedChannel(false)
    , mLeftMouseDown(false)
    , mRectVisible(true)
    , mMaskActive(false)
    , mSaveData(false)
    , mInitFrameBuffer(true)
    , mAKeyDown(false)
    , mBrushActive(false)
    , mShowPolygon(false)
    , mFrameBuffer(0)
    , mEditTex(0)
    , mEditDeltaTex(0)
    , mMaskTex(0)
    , mBeautyPassTex(0)
    , mRectSelection(0, 0, 0, 0)
    , mCurrentSelectionId(-1)
    , mCurrentSelectionOrderIndex(-1)
    , mCurrentMaskIndex(-1)
    , mBushSize(10)
    , mBrushValue(1.0)
{
    cout << "MultiPassImageView" << endl;
}

MultiPassImageView::~MultiPassImageView() {}

void MultiPassImageView::initTexture(ImageDataTypeVec textures)
{
    mTextures = std::move(textures);
    bindImage(mTextures[0].first.texture());
    initShader();
}

void MultiPassImageView::initPolygonMask(ImageDataType polyTexture)
{
    mPolyTexture = std::move(polyTexture);
    mShowPolygon = true;
}

void MultiPassImageView::initShader()
{
    cout << "initShader " << endl;

    string vertShaderStr = JWH::readFile("../shaders/multipass.vert");
    string fragShaderStr = JWH::readFile("../shaders/multipass.frag");

    string stringToReplace = "const int MAX_CHANNELS = XXX;";
    size_t pos = fragShaderStr.find(stringToReplace);

    string new_max_channels_string =
        "const int MAX_CHANNELS = " +
        to_string(std::max(1, mChannelEditsManager->noChannels())) + ";";

    fragShaderStr.replace(pos, stringToReplace.length(),
                          new_max_channels_string);

    mShader.init("MultiPassImageViewShader", vertShaderStr.c_str(),
                 fragShaderStr.c_str());

    MatrixXu indices(3, 2);
    indices.col(0) << 0, 1, 2;
    indices.col(1) << 2, 3, 1;

    MatrixXf vertices(2, 4);
    vertices.col(0) << 0, 0;
    vertices.col(1) << 1, 0;
    vertices.col(2) << 0, 1;
    vertices.col(3) << 1, 1;

    mShader.bind();
    mShader.uploadIndices(indices);
    mShader.uploadAttrib("vertex", vertices);
}

void MultiPassImageView::draw(NVGcontext *ctx)
{

    /*    if (mSaveData) {*/
    // saveImages();
    //// mSaveData = false;
    // return;
    /*}*/

    Widget::draw(ctx);
    nvgEndFrame(ctx);  // Flush the NanoVG draw stack, not necessary to call
                       // nvgBeginFrame afterwards.

    drawBrush(ctx);
    ImageView::drawImageBorder(ctx);
    if (mTextures.size() == 0) return;

    // Calculate several variables that need to be send to OpenGL in
    // order for
    // the
    // image to be
    // properly displayed inside the widget.
    const Screen *screen =
        dynamic_cast<const Screen *>(this->window()->parent());
    assert(screen);
    Vector2f screenSize = screen->size().cast<float>();
    Vector2f scaleFactor = mScale * imageSizeF().cwiseQuotient(screenSize);
    Vector2f positionInScreen = absolutePosition().cast<float>();
    Vector2f positionAfterOffset = positionInScreen + mOffset;
    Vector2f imagePosition = positionAfterOffset.cwiseQuotient(screenSize);

    glEnable(GL_SCISSOR_TEST);
    float r = screen->pixelRatio();
    glScissor(positionInScreen.x() * r,
              (screenSize.y() - positionInScreen.y() - size().y()) * r,
              size().x() * r, size().y() * r);

    mShader.bind();

    int texture_index_count = 0;

    // Add original rgb image
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mTextures[0].first.texture());  // RGB Texture
    mShader.setUniform("input_rgb", texture_index_count++);

    // Add render passes as 3D texture
    glActiveTexture(GL_TEXTURE0 + texture_index_count);
    glBindTexture(GL_TEXTURE_3D,
                  mTextures[1].first.texture());  // 3D Texture of Passes
    mShader.setUniform("render_passes", texture_index_count++);

    // Add single mask
    if (mMaskActive && validMaskSelected()) {

        ImageDataType &maskGLData = mChannelEditsManager->getMaskGLData(
            getSelectedChannelId(), mCurrentMaskIndex);
        glActiveTexture(GL_TEXTURE0 + texture_index_count);
        glBindTexture(GL_TEXTURE_2D,
                      maskGLData.first.texture());  // 3D Texture of Passes
        mShader.setUniform("mask", texture_index_count++);
    }

    // Add 3D Texture Of Masks
    for (auto i = 0; i < mChannelEditsManager->noChannels(); i++) {
        glActiveTexture(GL_TEXTURE0 + texture_index_count);
        Image3DDataType &mask3DTexture =
            mChannelEditsManager->getMask3DTexture(i);
        glBindTexture(GL_TEXTURE_3D,
                      mask3DTexture.first.texture());  // 3D Texture of Passes
        auto uniform_texture_string = "edit_masks[" + to_string(i) + "]";
        mShader.setUniform(uniform_texture_string, texture_index_count++);
    }

    static bool polyLoadedOnce = false;
    if (!polyLoadedOnce && mShowPolygon) polyLoadedOnce = true;

    if (polyLoadedOnce) {
        glActiveTexture(GL_TEXTURE0 + texture_index_count);
        glBindTexture(GL_TEXTURE_2D,
                      mPolyTexture.first.texture());  // RGB Texture
        mShader.setUniform("polygon_select", texture_index_count++);
    }

    // Add 2D Texture Array of Parameters
    glActiveTexture(GL_TEXTURE0 + texture_index_count);
    ImageData2DArray &param2DTexture =
        mChannelEditsManager->getEditParameters();
    glBindTexture(GL_TEXTURE_2D_ARRAY, param2DTexture.first.texture());

    mShader.setUniform("edit_params", texture_index_count++);

    if (mSaveData) {
        auto img_size = imageSize();
        glViewport(positionInScreen.x(),
                   (screenSize.y() - positionInScreen.y() - size().y()),
                   img_size(0), img_size(1));

        saveImages();
        return;
    }
    {
        Vector4f ss_rect = getScreenSpaceRect();

        for (auto i = 0; i < mSelectedChannels->size(); i++) {
            mShader.setUniform("selectedChannels[" + to_string(i) + "]",
                               int((*mSelectedChannels)[i]));

            // cout << int((*mSelectedChannels)[i]) << " ";
        }
        // cout << endl;

        mShader.setUniform("selectedRectangle", ss_rect);
        mShader.setUniform("currentSelectionIndex", getSelectedChannelId());
        mShader.setUniform("showOriginalRGB", int(mToggleRGBAndPasses));  //
        mShader.setUniform("showSelectedChannel", int(mShowSelectedChannel));
        mShader.setUniform("showMask", int(mToggleShowMask));
        mShader.setUniform("scaleFactor", scaleFactor);
        mShader.setUniform("position", imagePosition);
        mShader.setUniform("singleEditIndex", -1);
        mShader.setUniform("showPolygon", int(mShowPolygon));
        mShader.drawIndexed(GL_TRIANGLES, 0, 2);
    }
    glDisable(GL_SCISSOR_TEST);

    if (helpersVisible()) ImageView::drawHelpers(ctx);

    drawSelection(ctx);

    ImageView::drawWidgetBorder(ctx);
}

void MultiPassImageView::saveImages()
{

    cout << "saveImages " << endl;
    auto img_size = imageSize();
    if (mFrameBuffer) {
        glDeleteFramebuffers(1, &mFrameBuffer);
        mFrameBuffer = 0;
    }
    glGenFramebuffers(1, &mFrameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffer);

    {

        if (!mEditTex) {
            glGenTextures(1, &mEditTex);
        }

        // "Bind" the newly created texture : all future texture functions
        // will
        // modify this texture

        glBindTexture(GL_TEXTURE_2D, mEditTex);

        // Give an empty image to OpenGL ( the last "0" )
        glTexImage2D(GL_TEXTURE_2D, 0, glInternalFormat, img_size(0),
                     img_size(1), 0, glFormat, GLDataType, 0);
        // Poor filtering. Needed !
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, mEditTex, 0);
    }

    {

        if (!mEditDeltaTex) {
            glGenTextures(1, &mEditDeltaTex);
        }

        // "Bind" the newly created texture : all future texture functions
        // will
        // modify this texture

        glBindTexture(GL_TEXTURE_2D, mEditDeltaTex);

        // Give an empty image to OpenGL ( the last "0" )
        glTexImage2D(GL_TEXTURE_2D, 0, glInternalFormat, img_size(0),
                     img_size(1), 0, glFormat, GLDataType, 0);
        // Poor filtering. Needed !
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2,
                             mEditDeltaTex, 0);
    }

    {

        if (!mMaskTex) {
            glGenTextures(1, &mMaskTex);
        }

        // "Bind" the newly created texture : all future texture functions
        // will
        // modify this texture

        glBindTexture(GL_TEXTURE_2D, mMaskTex);

        // Give an empty image to OpenGL ( the last "0" )
        glTexImage2D(GL_TEXTURE_2D, 0, glInternalFormat, img_size(0),
                     img_size(1), 0, glFormat, GLDataType, 0);
        // Poor filtering. Needed !
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, mMaskTex, 0);
    }

    {

        if (!mBeautyPassTex) {
            glGenTextures(1, &mBeautyPassTex);
        }

        // "Bind" the newly created texture : all future texture functions
        // will
        // modify this texture

        glBindTexture(GL_TEXTURE_2D, mBeautyPassTex);

        // Give an empty image to OpenGL ( the last "0" )
        glTexImage2D(GL_TEXTURE_2D, 0, glInternalFormat, img_size(0),
                     img_size(1), 0, glFormat, GLDataType, 0);
        // Poor filtering. Needed !
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4,
                             mBeautyPassTex, 0);
    }

    // Set the list of draw buffers.
    GLenum DrawBuffers[5] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                             GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3,
                             GL_COLOR_ATTACHMENT4};
    glDrawBuffers(5, DrawBuffers);  // "1" is the size of DrawBuffers

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        assert(false);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffer);

    /// glViewport(0, 0, img_size(0), img_size(1));

    /*  glScissor(positionInScreen.x() * r,*/
    //(screenSize.y() - positionInScreen.y() - size().y()) * r,
    /*size().x() * r, size().y() * r);*/

    mShader.setUniform("selectedRectangle", Vector4f(0, 0, 1, 1));
    mShader.setUniform("showOriginalRGB", int(true));
    mShader.setUniform("scaleFactor", Vector2f(1, 1));
    mShader.setUniform("position", Vector2f(0, 0));
    mShader.setUniform("showPolygon", int(false));

    EditTransferScreen *screen =
        dynamic_cast<EditTransferScreen *>(this->window()->parent());

    vector<String> filenames = screen->getFilenames();

    auto found = filenames[0].find("_.");
    auto save_dir = filenames[0].substr(0, found) + "edits/FB";

    for (auto i = 0; i < mChannelEditsManager->noChannels(); i++) {
        for (auto j = 0; j < mChannelEditsManager->noChannelMasks(i); j++) {

            mShader.setUniform("currentSelectionIndex", i);
            mShader.setUniform("showSelectedChannel", int(true));
            mShader.setUniform("showMask", int(mToggleShowMask));
            mShader.setUniform("singleEditIndex", j);

            mShader.drawIndexed(GL_TRIANGLES, 0, 2);

            glBindFramebuffer(GL_READ_FRAMEBUFFER, mFrameBuffer);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mFrameBuffer);

            auto default_filename = save_dir + filenames[i + 1].substr(found);
            default_filename =
                default_filename.substr(0, default_filename.size() - 4);
            {
                cv::Mat edit(img_size(1), img_size(0), CV_32FC3,
                             Scalar(1, 1, 1));
                glBindTexture(GL_TEXTURE_2D, mEditTex);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GLDataType, edit.data);
                cv::flip(edit, edit, 0);

                string edit_filename;

                if (mWriteEditToTemp) {
                    edit_filename = filenames[0].substr(0, found) +
                                    "temp/Edit_" + to_string(j) + ".png";
                }
                else {
                    edit_filename =
                        default_filename + "_Edit_" + to_string(j) + ".exr";
                }

                cv::imwrite(edit_filename, edit);
            }

            if (!mWriteEditToTemp) {

                {
                    cv::Mat edit_delta(img_size(1), img_size(0), CV_32FC3,
                                       Scalar(0, 0, 0));
                    glBindTexture(GL_TEXTURE_2D, mEditDeltaTex);
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GLDataType,
                                  edit_delta.data);
                    cv::flip(edit_delta, edit_delta, 0);

                    auto delta_filename =
                        default_filename + "_Delta_" + to_string(j) + ".exr";
                    cv::imwrite(delta_filename, edit_delta);
                }

                {
                    cv::Mat edit_mask(img_size(1), img_size(0), CV_32FC3,
                                      Scalar(0));
                    glBindTexture(GL_TEXTURE_2D, mMaskTex);
                    glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GLDataType,
                                  edit_mask.data);
                    cv::flip(edit_mask, edit_mask, 0);
                    edit_mask.convertTo(edit_mask, CV_8UC3, 255.0);
                    Mat grey_edit_mask;
                    cv::cvtColor(edit_mask, grey_edit_mask, CV_BGR2GRAY);
                    auto mask_filename =
                        default_filename + "_Mask_" + to_string(j) + ".png";
                    cv::imwrite(mask_filename, grey_edit_mask);
                }
            }

/*            auto edit = mChannelEditsManager->getEdit(i, j);*/

            //// Blur all channels with blur params if needed
            //if (edit->getBlurSize() > 0) {
                //for (auto blur_index = 0;
                     //blur_index < mChannelEditsManager->noChannels();
                     //blur_index++) {
                    
                    //mShader.setUniform("currentSelectionIndex", blur_index);

                    //mShader.drawIndexed(GL_TRIANGLES, 0, 2);
                    //cv::Mat edit(img_size(1), img_size(0), CV_32FC3,
                                 //Scalar(1, 1, 1));
                    //glBindTexture(GL_TEXTURE_2D, mEditTex);
                    //glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GLDataType,
                                  //edit.data);
                    //cv::flip(edit, edit, 0);

                    //string edit_filename;

                    //if (mWriteEditToTemp) {
                        //edit_filename = filenames[0].substr(0, found) +
                                        //"temp/Blur_" + to_string(j) + "_" +
                                        //to_string(blur_index) + ".png";
                    //}
                    //else {
                        //edit_filename = default_filename + "_Blur_" +
                                        //to_string(j) + "_" +
                                        //to_string(blur_index) + ".exr";
                    //}

                    //cv::imwrite(edit_filename, edit);
                //}
            /*}*/

            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }
    }

    for (auto i = 0; i < mSelectedChannels->size(); i++) {
        mShader.setUniform("selectedChannels[" + to_string(i) + "]",
                           int((*mSelectedChannels)[i]));
        // cout << int((*mSelectedChannels)[i]) << " ";
    }

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mShader.setUniform("currentSelectionIndex", getSelectedChannelId());
    mShader.setUniform("showOriginalRGB", int(mToggleRGBAndPasses));  //
    mShader.setUniform("showSelectedChannel", int(mShowSelectedChannel));
    mShader.setUniform("showMask", int(mToggleShowMask));
    mShader.setUniform("singleEditIndex", -1);
    mShader.setUniform("showPolygon", int(mShowPolygon));
    mShader.drawIndexed(GL_TRIANGLES, 0, 2);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, mFrameBuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mFrameBuffer);

    mEditBeautyPassMat.create(img_size(1), img_size(0), CV_32FC3);

    glBindTexture(GL_TEXTURE_2D, mBeautyPassTex);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GLDataType,
                  mEditBeautyPassMat.data);
    cv::flip(mEditBeautyPassMat, mEditBeautyPassMat, 0);

    std::time_t now = std::time(NULL);
    std::tm *ptm = std::localtime(&now);
    char buffer[32];
    // Format: Mo, 15.06.2009 20:20:00
    std::strftime(buffer, 32, "%d.%m.%Y_%H:%M:%S", ptm);
    cv::imwrite(save_dir + buffer + "_Beauty.exr", mEditBeautyPassMat);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    /*    glDeleteRenderbuffers(1, &mFrameBuffer);*/
    /*glDeleteTextures(1, &mEditTex);*/

    if (!mWriteEditToTemp) writeParameters();
    mSaveData = false;
    mWriteEditToTemp = false;
}

bool MultiPassImageView::keyboardCharacterEvent(unsigned int codepoint)
{

    switch (codepoint) {
        case 't':  // Toggle Original Image and Current Edit
            mToggleRGBAndPasses = !mToggleRGBAndPasses;
            break;

        case 'r':
            initShader();
            break;
        case 'q':
            mShowPolygon = !mShowPolygon;
            mRectSelection.clear();
            break;
        /*        case 's':*/
        // mSaveData = true;
        /*break;*/
        case 'p':
            if (validMaskSelected()) {
                auto edit = mChannelEditsManager->getEdit(
                    getSelectedChannelId(), mCurrentMaskIndex);
                edit->toggleMaskParams();
                mChannelEditsManager->updateEditParametersTexture(
                    getSelectedChannelId());
            }
            break;
        /*        case 'q':*/
        // if (mCurrentSelectionOrderIndex > -1) {
        // if (--mCurrentSelectionOrderIndex == -1) {
        // mCurrentSelectionOrderIndex = mSelectionOrdering.size() - 1;
        //}
        //}
        // break;
        // case 'w':
        // if (mCurrentSelectionOrderIndex > -1) {
        // if (++mCurrentSelectionOrderIndex ==
        // mSelectionOrdering.size()) {
        // mCurrentSelectionOrderIndex = 0;
        //}
        //}
        /*break;*/
        default:
            return ImageView::keyboardCharacterEvent(codepoint);
    }
    return ImageView::keyboardCharacterEvent(codepoint);
}

bool MultiPassImageView::keyboardEvent(int key, int /*scancode*/, int action,
                                       int modifiers)
{

    // std::cout << key << " " << action << " " << modifiers << std::endl;
    // A key for viewing selected channel
    if (key == 65 && action == 1) {
        mShowSelectedChannel = true;
    }
    else if (key == 65 && action == 0) {

        mShowSelectedChannel = false;
    }
    else if (key == 83 && action == 1 && getSelectedChannelId() > -1) {
        mToggleShowMask = true;
        mMaskActive = true;
    }
    else if (key == 83 && action == 0) {

        mToggleShowMask = false;
        mShowSelectedChannel = false;
    }

    return ImageView::keyboardEvent(key, 0, action, modifiers);
}

bool MultiPassImageView::mouseButtonEvent(const Vector2i &p, int button,
                                          bool down, int modifiers)
{
    ImageView::mouseButtonEvent(p, button, down, modifiers);
    if (button == GLFW_MOUSE_BUTTON_1 && down && mBrushActive) {
        mLeftMouseDown = true;
        startMousePoint(0) = p(0);
        startMousePoint(1) = p(1);
        drawMask(p);
        mCursorPos = p;
    }
    else if (button == GLFW_MOUSE_BUTTON_1 && down &&
             (!mRectSelection.valid() || !channelSelected())) {
        auto img_pos = imageCoordinateAt(Vector2f(p[0], p[1]));
        mRectSelection.setPoint(img_pos);
        mRectSelection.setWidthHeight(img_pos + Vector2f(1, 1));
        mCurrentSelectionOrderIndex = -1;
        mToggleShowMask = false;
        mLeftMouseDown = true;
    }
    else if (button == GLFW_MOUSE_BUTTON_1 && down &&
             (mRectSelection.valid() || channelSelected())) {
        // clearCurrentSelection();
        mRectSelection.clear();
    }
    else if (button == GLFW_MOUSE_BUTTON_1 && !down && mRectSelection.valid() &&
             !mBrushActive) {
        cout << "Find Channel " << endl;
        mLeftMouseDown = false;
        selectChannel();
    }
    else if (button == GLFW_MOUSE_BUTTON_1 && !down && mBrushActive) {
        mLeftMouseDown = false;
        startMousePoint(0) = 0;
        startMousePoint(1) = 0;
        mCursorPos = p;

        EditTransferScreen *screen =
            dynamic_cast<EditTransferScreen *>(this->window()->parent());
        screen->updateMaskIcon(mCurrentSelectionId, mCurrentMaskIndex);
    }

    requestFocus();
    return true;
}

bool MultiPassImageView::mouseDragEvent(const Vector2i &p, const Vector2i &rel,
                                        int button, int modifiers)
{
    if (button == 1 /*Left mouse button*/ && mLeftMouseDown && !mBrushActive) {
        mRectSelection.setWidthHeight(imageCoordinateAt(Vector2f(p[0], p[1])));
        return true;
    }
    else if (button == 1 /*Left mouse button*/ && mLeftMouseDown &&
             mBrushActive) {
        drawMask(p);
        startMousePoint(0) = p(0);
        startMousePoint(1) = p(1);
        mCursorPos = p;
        return true;
    }

    return ImageView::mouseDragEvent(p, rel, button, modifiers);
}
bool MultiPassImageView::scrollEvent(const Vector2i &p, const Vector2f &rel)
{
    return ImageView::scrollEvent(p, rel);
}

bool MultiPassImageView::mouseMotionEvent(const Vector2i &p,
                                          const Vector2i &rel, int button,
                                          int modifiers)
{
    mCursorPos = p;
    return true;
}

void MultiPassImageView::drawSelection(NVGcontext *ctx)
{
    if (!mRectSelection.valid() || !mRectVisible) return;

    auto image_coordinate =
        positionForCoordinate(mRectSelection.getPosVector());

    auto width_height_scaled = mRectSelection.getWHVector() * mScale;

    nvgBeginPath(ctx);
    nvgRect(ctx, image_coordinate.x(), image_coordinate.y(),
            width_height_scaled[0], width_height_scaled[1]);
    nvgStrokeColor(ctx, nvgRGB(255, 0, 0));
    nvgStrokeWidth(ctx, 1);
    nvgStroke(ctx);
    // cout << "drawSelection " << endl;
}

void MultiPassImageView::drawBrush(NVGcontext *ctx)
{
    if (!mBrushActive) return;

    nvgBeginPath(ctx);
    nvgCircle(ctx, mCursorPos[0], mCursorPos[1], mBushSize / 2);
    nvgStrokeColor(ctx, nvgRGB(175, 175, 255));
    nvgStrokeWidth(ctx, 1);
    nvgStroke(ctx);
    // cout << "drawSelection " << endl;
}

void MultiPassImageView::drawMask(const Vector2i &p)
{
    auto edit =
        mChannelEditsManager->getEdit(mCurrentSelectionId, mCurrentMaskIndex);

    edit->drawMask(
        imageCoordinateAt(Vector2f(startMousePoint[0], startMousePoint[1])),
        imageCoordinateAt(Vector2f(p[0], p[1])), mBushSize, mBrushValue);

    edit->setUpdateTexture(true);

    mChannelEditsManager->updateMasks(edit->getChannelIndexes());
}

void MultiPassImageView::performLayout(NVGcontext *ctx)
{
    Widget::performLayout(ctx);
}

int MultiPassImageView::getSelectedChannelId()
{
    if (mCurrentSelectionOrderIndex == -1)
        return mCurrentSelectionId;  // This would have been set from mask
                                     // selection tool
    else
        return mSelectionOrdering[mCurrentSelectionOrderIndex];
}

vector<int> MultiPassImageView::getSelectedChannelIds()
{
    vector<int> selectedIds;
    for (auto i = 0; i < mSelectedChannels->size(); i++) {
        if ((*mSelectedChannels)[i]) {
            selectedIds.push_back(i);
        }
    }
    return selectedIds;
}

bool MultiPassImageView::validMaskSelected()
{
    if (!channelSelected())
        return false;
    else if (mCurrentSelectionOrderIndex > -1)
        return mChannelEditsManager->channelHasMask(
            mSelectionOrdering[mCurrentSelectionOrderIndex]);
    else
        return mChannelEditsManager->channelHasMask(mCurrentSelectionId);
}

void MultiPassImageView::clearCurrentSelection()
{
    mCurrentSelectionOrderIndex = -1;
    mCurrentMaskIndex = -1;
    mCurrentSelectionId = -1;
    mToggleShowMask = false;
    mMaskActive = false;
    mShowSelectedChannel = false;
    mRectSelection.clear();
    initShader();
}

void MultiPassImageView::showMask()
{
    mToggleShowMask = true;
    mMaskActive = true;
}

void MultiPassImageView::writeParameters()
{
    cout << "writeParameters" << endl;
    EditTransferScreen *screen =
        dynamic_cast<EditTransferScreen *>(this->window()->parent());

    vector<String> filenames = screen->getFilenames();

    vector<Mat> imageMats = screen->getImageMats();
    Mat beautyPass;
    makeBeautyPassRGB(beautyPass, imageMats);

    /*    imshow("Lab", beautyPass);*/
    // Mat beautyPassRGB;
    // cvtColor(beautyPass, beautyPassRGB, CV_Lab2BGR);
    // imshow("RGB", beautyPassRGB);
    /*waitKey();*/

    cout << " MultiPassImageView::writeParameters() needs to be implemented "
            "again with new channel selectior"
         << endl;

    auto found = filenames[0].find("_.");
    auto save_dir = filenames[0].substr(0, found) + "edits/FB";

    for (auto i = 0; i < mChannelEditsManager->noChannels(); i++) {
        for (auto j = 0; j < mChannelEditsManager->noChannelMasks(i); j++) {
            auto edit = mChannelEditsManager->getEdit(i, j);

            Mat mask = edit->getMask();
            /*            cout << "selectRelativeChangePatches" << endl;*/
            // vector<Mat> labels;
            // mChannelSelector->selectRelativeChangePatches(
            /*mask, beautyPass, mEditBeautyPassMat, labels);*/

            /*            cout << "computeRelativePositioning" << endl;*/
            // edit->computeRelativePositioning(beautyPass,
            // mEditBeautyPassMat,
            /*labels);*/

            // edit.computeEditRatio(imageMats[i + 1]);

            // EditEmbedding::computeNeightbourStatistics(beautyPass, edit);

            //// imshow("Image mat", imageMats[i + 1]);
            //// waitKey();

            // Mat embedding =
            // EditEmbedding::computeEmbedding(edit_means, channelStats);

            auto default_filename = save_dir + filenames[i + 1].substr(found);
            default_filename =
                default_filename.substr(0, default_filename.size() - 4);
            auto params_filename =
                default_filename + "_Para_" + to_string(j) + ".yml";

            edit->writeParameters(params_filename);
        }
    }
}

// Private functions
void MultiPassImageView::selectChannel(Mat *polygon)
{
    static JWH::Rectangle previous(0, 0, 0, 0);

    auto img_size = imageSize();

    Mat boxPoly(img_size(1), img_size(0), CV_8UC3, Scalar(0, 0, 0));
    if (polygon == nullptr) {
        if (!mRectSelection.valid() || mRectSelection.same(previous) ||
            mRectSelection.area() < 20.0) {
            cout << "Rect not valid" << endl;
            mRectSelection.clear();
            return;
        }
        auto rect = JWH::convertToCVRect(mRectSelection, boxPoly);
        rectangle(boxPoly, rect, Scalar(255, 255, 255), -1);
    }

    EditTransferScreen *screen =
        dynamic_cast<EditTransferScreen *>(this->window()->parent());

    if (polygon == nullptr) {
        screen->selectChannel(&boxPoly, mSelectionOrdering);
        boxPoly.copyTo(polygonMask);
    }
    else {
        screen->selectChannel(polygon, mSelectionOrdering);
        (*polygon).copyTo(polygonMask);
    }
    mCurrentSelectionOrderIndex = 0;
    mShowSelectedChannel = false;
    mCurrentSelectionId = -1;

    previous = mRectSelection;
}

Vector4f MultiPassImageView::getScreenSpaceRect()
{
    auto ss_rect = Vector4f(0, 0, 1, 1);
    if (channelSelected()) {  //&&(mShowSelectedChannel || mToggleShowMask)

        ss_rect = mRectSelection.getRect();
        // Normalize for screen coordinates
        auto img_size = imageSizeF();
        ss_rect[0] /= img_size[0];
        ss_rect[2] /= img_size[0];
        ss_rect[1] /= img_size[1];
        ss_rect[3] /= img_size[1];

        if (!mRectVisible) ss_rect = Vector4f(0, 0, 1, 1);
    }
    return ss_rect;
}

void MultiPassImageView::setBrushState(bool state)
{
    mBrushActive = state;

    if (mBrushActive) {
        setCursor(Cursor::Crosshair);
    }
    else {
        setCursor(Cursor::Arrow);
    }
}
