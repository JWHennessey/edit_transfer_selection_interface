#ifndef EDIT_TRANSFER_SCREEN_H
#define EDIT_TRANSFER_SCREEN_H

#include <iostream>

#include <nanogui/button.h>
#include <nanogui/checkbox.h>
#include <nanogui/colorwheel.h>
#include <nanogui/combobox.h>
#include <nanogui/glutil.h>
#include <nanogui/graph.h>
#include <nanogui/imagepanel.h>
#include <nanogui/imageview.h>
#include <nanogui/label.h>
#include <nanogui/layout.h>
#include <nanogui/messagedialog.h>
#include <nanogui/opengl.h>
#include <nanogui/popupbutton.h>
#include <nanogui/screen.h>
#include <nanogui/slider.h>
#include <nanogui/tabwidget.h>
#include <nanogui/textbox.h>
#include <nanogui/theme.h>
#include <nanogui/toolbutton.h>
#include <nanogui/vscrollpanel.h>
#include <nanogui/window.h>
#include <opencv2/core.hpp>

#include "ChannelEditsManager.h"
#include "ChannelStat.h"
#include "GLTexture.h"
#include "MaskEdit.h"
#include "MultiPassImageView.h"
#include "Segmentor.h"
#include "jwh_util.h"

using namespace std;
using namespace Eigen;
using namespace cv;
using nanogui::Screen;
using nanogui::Window;
using nanogui::GroupLayout;
using nanogui::GridLayout;
using nanogui::BoxLayout;
using nanogui::ImageView;
using nanogui::Theme;
using nanogui::Button;
using nanogui::CheckBox;
using nanogui::Label;
using nanogui::Graph;
using nanogui::Slider;
using nanogui::ComboBox;
using nanogui::ToolButton;
using nanogui::file_dialog;
using nanogui::TabWidget;
using nanogui::ImagePanel;
using nanogui::VScrollPanel;
using nanogui::IntBox;
using nanogui::PopupButton;
using nanogui::Popup;
using nanogui::ColorWheel;
using nanogui::Color;
using nanogui::TextBox;
using nanogui::FloatBox;

enum MaskInitType { Auto, Empty };

class EditTransferScreen : public Screen {
    friend MultiPassImageView;

   public:
    EditTransferScreen(const Vector2i &size, const std::string &caption);
    bool resizeCallbackEvent(int x, int y) override;
    /*    vector<EditStat> computeChannelStatistics(Mat edit_mask, int
     * channel_index,*/
    /*int material_index);*/
    // vector<ChannelStat> computeChannelStatistics(Mat edit_mask);
    inline vector<String> getFilenames() { return mFilenames; }
    inline vector<Mat> getImageMats() { return mImageMats; }
    bool keyboardEvent(int key, int /*scancode*/, int action,
                       int modifiers) override;

   private:
    // Variables
    ImageDataTypeVec mImagesData;
    ImageDataType mPolygonData;
    vector<String> mFilenames;
    vector<Mat> mImageMats;       // .exr images, first image is input RGB
    vector<Mat> mMaterialIdMats;  // .exr images, first image is input RGB
    vector<pair<int, string>> imageIcons;
    vector<pair<int, string>> editIcons;
    vector<pair<float, int>> mChannelOrderingScore;
    vector<bool> mSelectedChannels;
    shared_ptr<ChannelEditsManager> mChannelEditsManager;
    shared_ptr<ChannelSelector> mChannelSelector;
    vector<shared_ptr<MaskEdit>> mMaskEdits;
    shared_ptr<MaskEdit> mSelectedEdit;
    float mPrevHistMinValue;
    float mPrevHistMaxValue;
    String mFolder;
    int mSelectedMaterialId;
    int mselectedEditIcon;

    // Window variables, not unique for simplicity with nanogui
    Window *imageWindow;
    Window *controlsWindow;
    TabWidget *parametersControls;
    Window *tabsWindow;
    Widget *editsLayer;
    ImagePanel *channelIconsImgPanel;
    ImagePanel *editIconsImgPanel;
    MultiPassImageView *imageView;
    // Label *selectedChannelLabel;
    Button *showCompositeToggleBtn;
    Button *showChannelToggleBtn;
    Button *showMaskToggleBtn;
    Button *removeBtn;
    Graph *graph;
    Slider *gammaRGBSlider;
    Slider *minHistRGBSlider;
    Slider *maxHistRGBSlider;
    Slider *gammaRSlider;
    Slider *minHistRSlider;
    Slider *maxHistRSlider;
    Slider *gammaGSlider;
    Slider *minHistGSlider;
    Slider *maxHistGSlider;
    Slider *gammaBSlider;
    Slider *minHistBSlider;
    Slider *maxHistBSlider;
    Slider *minOutRGBSlider;
    Slider *maxOutRGBSlider;
    Slider *minOutRSlider;
    Slider *maxOutRSlider;
    Slider *minOutGSlider;
    Slider *maxOutGSlider;
    Slider *minOutBSlider;
    Slider *maxOutBSlider;
    Slider *exposureSlider;
    Slider *hueSlider;
    Slider *saturationSlider;
    Slider *lightnessSlider;
    Slider *brightnessSlider;
    Slider *contrastSlider;
    CheckBox *applyMaskCheckBox;
    CheckBox *rectVisibleCheckbox;
    // Functions
    void resizeWidgets(bool fitImageView = true);
    void updateTheme();
    void loadImages(String folder, bool loadTransfer = false);
    void loadEdits(String folder, int filename_index,
                   bool loadTransfer = false);
    void loadPolygon(String filename);
    void loadMaterialIds(String folder);
    void initChannelSelector();
    void initChannelEdits();
    void initControlsWindow();
    void initMovablePanel();
    void initEditParmWidgets();
    void initVisibilityWidgets();
    void updateGraph();
    void selectChannel(Mat *roi_mask, vector<int> &ordering);
    void createMask(MaskInitType initType);
    void autoMaskRegion();
    void showParamsWidget();
    void selectMask(int index);

    void updateMaskIcon(int selectedChannel, int maskIndex);
};

#endif
