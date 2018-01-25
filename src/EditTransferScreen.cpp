#include "EditTransferScreen.h"

#include <iomanip>

EditTransferScreen::EditTransferScreen(const Vector2i &size,
                                       const std::string &caption)
    : Screen(size, caption)
    , mChannelEditsManager(new ChannelEditsManager())
    , mChannelSelector(new ChannelSelector())
    , mSelectedEdit(nullptr)
    , mPrevHistMinValue(0.0)
    , mPrevHistMaxValue(1.0)
    , mSelectedMaterialId(-1)
    , mselectedEditIcon(0)
{

    updateTheme();
    setLayout(new BoxLayout(nanogui::Orientation::Horizontal,
                            nanogui::Alignment::Fill, 0, 0));
    // setLayout(new GroupLayout());

    imageWindow = new Window(this, "");
    imageWindow->setId("ImageWindow");
    imageView = new MultiPassImageView(imageWindow, mChannelEditsManager,
                                       mChannelSelector, &mSelectedChannels);
    imageView->setGridThreshold(10);
    imageView->setPixelInfoThreshold(10);
    imageView->setPixelInfoCallback(
        [this](const Vector2i &index) -> pair<string, nanogui::Color> {
            /*        auto &imageData = mImagesData[0].second;*/
            /*auto &textureSize = imageView->imageSize();*/
            string stringData("String");
            /*        uint16_t channelSum = 0;*/
            // for (int i = 0; i != 4; ++i) {
            // auto &channelData =
            // imageData[4 * index.y() * textureSize.x() + 4 * index.x() + i];
            // channelSum += channelData;
            // stringData += (to_string(static_cast<int>(channelData)) + "\n");
            //}
            // float intensity = static_cast<float>(255 - (channelSum / 4)) /
            // 255.0f;
            // float colorScale =
            /*intensity > 0.5f ? (intensity + 1) / 2 : intensity / 2;*/
            nanogui::Color textColor = nanogui::Color(0.5, 0.5, 0.5, 1.0f);
            return {stringData, textColor};
        });

    initControlsWindow();
    // initMovablePanel();

    resizeWidgets();
    parametersControls->setVisible(false);

    /*        "/Users/JamesHennessey/Projects/edit_transfer_interface/scenes/"*/
    /*"beer_bottles/budwiser/",*/

    // So I don't need to manually open file each time
    /*    loadImages(*/
    //"/Users/JamesHennessey/Projects/edit_transfer_interface/scenes/"
    //// "beer_bottles/budwiser/",
    ////"siggraph_text/view1/",
    //// "beer_bottles/grimbergen_dof_view1/",
    //"car/view1/",
    /*false);*/
}

bool EditTransferScreen::resizeCallbackEvent(int x, int y)
{
    resizeWidgets();
    return Screen::resizeCallbackEvent(x, y);
}

// Private
void EditTransferScreen::resizeWidgets(bool fitImageView)
{
    // cout << width() << " " << height() << endl;
    // cout << "resizeWidgets()" << endl;
    float new_main_width = (float)width() * 0.7f;
    float new_secondary_width = (float)width() * 0.3f;

    // if (fitImageView) {

    imageWindow->setPosition(Vector2i(0, 0));
    imageWindow->setLayout(new GroupLayout(0, 0, 0, 0));
    imageWindow->setFixedSize(Vector2i(new_main_width, height()));
    imageWindow->setSize(Vector2i(new_main_width, height()));

    imageView->setLayout(new GroupLayout(0, 0, 0, 0));

    imageView->setFixedSize(Vector2i(new_main_width, height()));
    imageView->setSize(Vector2i(new_main_width, height()));
    imageView->fit();
    //}

    controlsWindow->setPosition(Vector2i(new_main_width, 0));
    controlsWindow->setFixedSize(Vector2i(new_secondary_width, height()));
    controlsWindow->setSize(Vector2i(new_secondary_width, height()));

    performLayout();
}

void EditTransferScreen::updateTheme()
{
    auto *windowtheme = theme();
    windowtheme->mWindowDropShadowSize = 0;
    windowtheme->mWindowFillUnfocused = windowtheme->mWindowFillFocused;
    windowtheme->mWindowTitleUnfocused = windowtheme->mWindowTitleFocused;
    setTheme(windowtheme);
}

void EditTransferScreen::loadImages(String folder, bool loadTransfer)
{
    mFolder = folder;
    mImagesData.clear();
    mImageMats.clear();
    mMaterialIdMats.clear();
    cout << folder << endl;
    glob(folder, mFilenames);

    Mat rgb_image;  // Read RGB Into own mat but add to vec later

    string png_folder = folder + "png";
    // cout << png_folder << endl;
    imageIcons = nanogui::loadImageDirectory(mNVGContext, png_folder);

    for (auto i = 0; i < mFilenames.size(); i++) {
        auto file = mFilenames[i];
        // auto found_exr = file.find(".exr");
        auto found_rgb = file.find("_.exr");
        auto found_exr = file.find(".exr");
        if (found_rgb != string::npos) {  // If is beauty flass
            GLTexture texture("RGB");
            rgb_image = imread(file, cv::IMREAD_UNCHANGED);

            auto data = texture.load(rgb_image);
            mImagesData.emplace_back(std::move(texture), std::move(data));

            mFilenames.erase(mFilenames.begin() + i);      // Delete filename
            mFilenames.emplace(mFilenames.begin(), file);  // then add at start
        }
        else if (found_exr == string::npos) {  // If not an image e.g. .DS_Store

            mFilenames.erase(mFilenames.begin() + i--);
        }
        /*        else {  // All other images*/
        // Mat image = imread(file, cv::IMREAD_UNCHANGED);
        // mImageMats.push_back(image);
        //// cout << file << endl;
        /*}*/
    }

    mFilenames.clear();
    mFilenames.push_back(mFolder + "/_.exr");

    // cout << "mFilenames.front()  " << mFilenames.front() << endl;

    int icon_count = 0;
    mChannelOrderingScore.clear();
    for (auto icon : imageIcons) {
        cout << icon.first << " " << icon.second << endl;
        auto png_section = icon.second.find("/png/");
        string file = icon.second.substr(png_section + 5);
        string exr_filename = folder + file + ".exr";
        Mat image = imread(exr_filename, cv::IMREAD_UNCHANGED);
        mImageMats.push_back(image);
        mFilenames.push_back(exr_filename);
        /*        cout << "exr_filename" << exr_filename << endl;*/
        // imshow("image", image);
        /*waitKey();*/
        mChannelOrderingScore.push_back(make_pair(0.0, icon_count++));
        mSelectedChannels.push_back(false);

        Mat img_grey;
        cvtColor(image, img_grey, CV_RGB2GRAY);

        img_grey.convertTo(img_grey, CV_32FC1);

        imwrite(
            "/Users/JamesHennessey/Projects/edit_transfer_interface/scenes/"
            "Grey_" +
                file + ".exr",
            img_grey);

        // double min, max;
        // cv::minMaxLoc(image, );
    }

    // Read other images into 3D texture
    GL3DTexture texture("RenderPasses");
    auto data = texture.load(mImageMats);
    mImagesData.emplace_back(std::move(texture), std::move(data));

    mImageMats.emplace(mImageMats.begin(),
                       rgb_image);  // Add rgbimages at start

    assert(mImagesData.size() > 0);

    cout << "imageIcons.size() " << imageIcons.size() << endl;
    cout << "mImageMats.size() " << mImageMats.size() << endl;

    vector<bool> selected(imageIcons.size());

    channelIconsImgPanel->setImages(imageIcons);
    channelIconsImgPanel->setSelected(selected);

    /*    Mat newLayerEdit(mImageMats[0].cols, mImageMats[1].rows, CV_8UC4,*/
    /*Scalar(33, 33, 33, 255));*/

    Mat newLayerEdit(1000, 1000, CV_8UC4, Scalar(33, 33, 33, 255));

    putText(newLayerEdit, "New", Point(50, 400), 0, 15,
            Scalar(148, 148, 148, 255), 24);
    putText(newLayerEdit, "+", Point(400, 800), 0, 15,
            Scalar(148, 148, 148, 255), 24);

    /*    int img_id = nvgCreateImageRGBA(mNVGContext, mImageMats[0].rows,*/
    /*mImageMats[0].cols, 0, newLayerEdit.data);*/

    int img_id =
        nvgCreateImageRGBA(mNVGContext, 1000, 1000, 0, newLayerEdit.data);

    editIcons.push_back(
        make_pair(img_id, "mask" + to_string(editIcons.size())));

    editIconsImgPanel->setImages(editIcons);
    editIconsImgPanel->setSelected({false});

    imageView->initTexture(move(mImagesData));

    loadMaterialIds(folder);

    initChannelSelector();
    initChannelEdits();

    // computeChannelStatistics(folder);
    if (!loadTransfer) {
        folder = folder + "edits/";
    }
    else {
        folder = folder + "transfer_from/";
    }
    for (auto i = 1; i < mFilenames.size(); i++) {
        loadEdits(folder, i, loadTransfer);
    }

    resizeWidgets(false);
    imageView->initShader();
}

void EditTransferScreen::loadEdits(String folder, int filename_index,
                                   bool loadTransfer)
{
    auto input_filename = mFilenames[filename_index];
    auto found = input_filename.find("view");
    cout << "input_filename " << input_filename << endl;
    auto find_exr = input_filename.find("/_.exr");
    if (find_exr != string::npos) return;

    auto filename = input_filename.substr(found + 6);
    filename = filename.substr(0, filename.size() - 4);
    filename = filename.substr(2);
    std::cout << "loadEdits " << std::endl;
    vector<String> editFilenames;
    auto search_string = filename + "_Mask_";
    glob(folder, editFilenames);
    for (auto i = 0; i < editFilenames.size(); i++) {
        std::string edit_file = editFilenames[i];
        if (edit_file.find("_.exr") != string::npos ||
            edit_file.find(".DS_Store") != string::npos)
            continue;
        cout << "Edit File " << edit_file << endl;
        cout << "search_string " << search_string << endl;
        auto found = edit_file.find(search_string);
        cout << "Found mask " << found << endl;
        if (found != string::npos) {
            cout << "Opening " << edit_file << endl;
            Mat mask = imread(edit_file, CV_LOAD_IMAGE_COLOR);

            mask.convertTo(mask, CV_32FC3, 1.0 / 255.0);

            auto params_file = edit_file.substr(0, edit_file.size() - 4);
            found = params_file.find("Mask");
            params_file[found] = 'P';
            params_file[found + 1] = 'a';
            params_file[found + 2] = 'r';
            params_file[found + 3] = 'a';
            params_file = params_file + ".yml";

            cout << params_file << endl;
            cv::FileStorage file(params_file, cv::FileStorage::READ);
            cv::FileNode p = file["Params"];
            cv::Mat params;
            p >> params;

            int materialIndex = params.at<float>(0, 12);

            params_file = params_file.substr(0, params_file.size() - 4);
            int noRelativePatches = params.at<float>(0, 13);

            // cout << "noRelativePatches " << noRelativePatches << endl;
            vector<Mat> relativePatches;
            for (auto x = 0; x < noRelativePatches; x++) {
                auto patchFilename =
                    params_file + "_RelativePatch" + to_string(x) + ".png";
                Mat patch = imread(patchFilename, CV_LOAD_IMAGE_COLOR);
                patch.convertTo(patch, CV_32FC3, 1.0 / 255.0);
                relativePatches.push_back(patch);
                /*                imshow("patch", patch);*/
                /*waitKey();*/
            }

            MaskEdit mask_edit(mask, input_filename, params,
                               mImageMats[filename_index], Mat(), materialIndex,
                               relativePatches);

            vector<int> channel_indexs = mask_edit.getChannelIndexes();
            Mat maskFromEdit = mask_edit.getMask();

            int selectedMaskId = mChannelEditsManager->push_back(
                channel_indexs, move(mask_edit));

            auto tempEdit = mChannelEditsManager->getEdit(channel_indexs[0],
                                                          selectedMaskId);

            mMaskEdits.push_back(tempEdit);

            Mat rgbaMask(maskFromEdit.rows, maskFromEdit.cols, CV_8UC4);

            cvtColor(maskFromEdit, rgbaMask, CV_RGB2RGBA, 4);
            rgbaMask.convertTo(rgbaMask, CV_8UC4, 255);

            int img_id =
                nvgCreateImageRGBA(mNVGContext, maskFromEdit.rows,
                                   maskFromEdit.cols, 0, rgbaMask.data);

            editIcons.push_back(
                make_pair(img_id, "mask" + to_string(editIcons.size())));

            if (loadTransfer) {
                std::cout << "Can't load transfers " << std::endl;
                assert(false);
                /*                vector<ChannelStat> channelStats =*/
                // computeChannelStatistics(mask);

                // pair<Eigen::Vector3f, Eigen::Vector3f> edit_means =
                // mask_edit.getEditMean(mImageMats[i + 1]);

                // Mat embedding =
                /*EditEmbedding::computeEmbedding(edit_means, channelStats);*/

                mask_edit.transferParameters(mImageMats);
            }

            // vector<int> channelIds = {filename_index - 1};
            // mChannelEditsManager->push_back(channelIds, move(mask_edit));
        }
    }
    imageView->setSelectedChannelId(-1);
    imageView->setSelectedMaskId(-1);

    editIconsImgPanel->setImages(editIcons);
    vector<bool> selectedEdit(editIcons.size());
    selectedEdit[0] = true;
    editIconsImgPanel->setSelected(selectedEdit);
    cout << "editIcons.size() " << editIcons.size() << endl;

    resizeWidgets(false);
}

void EditTransferScreen::loadMaterialIds(String folder)
{
    folder = folder + "material_ids/";

    vector<String> materialFilenames;
    glob(folder, materialFilenames);
    for (auto materialFile : materialFilenames) {
        auto found = materialFile.find(".png");
        if (found != string::npos) {
            cout << materialFile << endl;
            Mat material = imread(materialFile, CV_LOAD_IMAGE_COLOR);
            cvtColor(material, material, COLOR_BGR2GRAY);
            threshold(material, material, 10, 255, THRESH_BINARY);
            mMaterialIdMats.push_back(material);
        }
    }
}

/*vector<EditStat> EditTransferScreen::computeChannelStatistics(*/
// Mat edit_mask, int channel_index, int material_index)
//{
// vector<EditStat> mChannelStats;
// for (auto i = 0; i < mFilenames.size(); i++) {
// cout << mFilenames[i] << endl;
// EditStat neighbourStats = EditEmbedding::computeNeightbourStatistics(
// mImageMats[], edit_mask, mMaterialIdMats[material_index]);
// mChannelStats.push_back(neighbourStats);
//}
// return mChannelStats;
/*}*/

void EditTransferScreen::initChannelSelector()
{
    mChannelSelector->create(make_shared<vector<String>>(mFilenames),
                             make_shared<vector<Mat>>(mImageMats),
                             make_shared<vector<Mat>>(mMaterialIdMats));
}

void EditTransferScreen::initChannelEdits()
{
    mChannelEditsManager->init(mImageMats.size() - 1);
}

void EditTransferScreen::loadPolygon(String filename)
{
    GLTexture texture("Polygon");
    cv::Mat poly_image = imread(filename, cv::IMREAD_COLOR);

    Mat contours;
    cvtColor(poly_image, contours, COLOR_BGR2GRAY);

    cv::threshold(poly_image, poly_image, 10, 255, THRESH_BINARY);

    int thresh = 100;
    Canny(contours, contours, thresh, thresh * 2, 3);
    dilate(contours, contours, Mat(), Point(-1, -1), 2, 1, 1);
    std::vector<cv::Mat> images(3);
    Mat black = Mat::zeros(contours.rows, contours.cols, CV_8UC1);
    images.at(0) = black;  // for blue channel
    images.at(1) = black;  // for green channel
    images.at(2) = contours;
    cv::Mat polyBoundry;
    cv::merge(images, polyBoundry);

    polyBoundry.convertTo(polyBoundry, CV_32FC3, 1.0 / 255.0);

    auto data = texture.load(polyBoundry);
    mPolygonData = make_pair(std::move(texture), std::move(data));
    imageView->initPolygonMask(move(mPolygonData));

    imageView->selectChannel(&poly_image);
}

void EditTransferScreen::initControlsWindow()
{
    controlsWindow = new Window(this, "");
    controlsWindow->setId("ControlsWindow");
    controlsWindow->setLayout(new BoxLayout(nanogui::Orientation::Vertical,
                                            nanogui::Alignment::Fill, 5, 5));

    Window *selectionPanel = new Window(controlsWindow, "");  //
    selectionPanel->setLayout(new BoxLayout(nanogui::Orientation::Vertical,
                                            nanogui::Alignment::Fill, 5, 5));

    Window *selectionPanelButtons = new Window(selectionPanel, "");

    /*    GridLayout *layout = new GridLayout(nanogui::Orientation::Horizontal,
     * 4,*/
    // nanogui::Alignment::Fill, 5, 5);
    // layout->setColAlignment(
    //{nanogui::Alignment::Maximum, nanogui::Alignment::Fill});
    /*layout->setSpacing(0, 0);*/

    selectionPanelButtons->setLayout(new BoxLayout(
        nanogui::Orientation::Horizontal, nanogui::Alignment::Fill, 5, 5));

    /*    fileButtons->setLayout(new
     * BoxLayout(nanogui::Orientation::Horizontal,*/
    /*nanogui::Alignment::Fill, 5, 5));*/

    ToolButton *b = new ToolButton(selectionPanelButtons, ENTYPO_ICON_FOLDER);
    b->setFlags(Button::NormalButton);
    b->setCallback([this] {

        auto filename = file_dialog({{"txt", "Text file"}}, false);

        auto found = filename.find("_.");
        if (found != string::npos) {
            auto folder = filename.substr(0, found);
            cout << folder << endl;
            loadImages(folder);
        }
    });

    b = new ToolButton(selectionPanelButtons, ENTYPO_ICON_SAVE);
    b->setFlags(Button::NormalButton);
    b->setCallback([this] {
        /*        cout << "File dialog result: "*/
        //<< file_dialog({{"png", "Portable Network Graphics"},
        //{"txt", "Text file"}},
        // true)
        /*<< endl;*/
        cout << "ToDo: Set custom save folder " << endl;
        imageView->mSaveData = true;
        // imageView->writeParameters();
    });

    b = new ToolButton(selectionPanelButtons, ENTYPO_ICON_FORWARD);
    b->setFlags(Button::NormalButton);
    b->setCallback([this] {
        auto filename = file_dialog({{"txt", "Text file"}}, false);
        auto found = filename.find("_.");
        if (found != string::npos) {
            auto folder = filename.substr(0, found);
            cout << folder << endl;
            loadImages(folder, true);
        }
    });

    b = new ToolButton(selectionPanelButtons, ENTYPO_ICON_LANDSCAPE_DOC);
    b->setFlags(Button::NormalButton);
    b->setCallback([this] {
        auto filename = file_dialog({{"png", "Image file"}}, false);
        auto found = filename.find(".png");
        if (found != string::npos) {
            loadPolygon(filename);
        }
    });

    /*    new Label(selectionPanelButtons, "Select:  ", "sans-bold");*/

    // Window *selectionButtons = new Window(selectionPanelButtons, "");

    // selectionButtons->setLayout(new
    // BoxLayout(nanogui::Orientation::Horizontal,
    // nanogui::Alignment::Fill, 5, 5));

    // b = new ToolButton(selectionButtons, ENTYPO_ICON_SEARCH);
    // b->setFlags(Button::NormalButton);
    // b->setCallback([this] {
    // if (imageView->rectSelectionValid()) {
    // imageView->selectChannel();
    //}
    // else {
    // auto dlg = new nanogui::MessageDialog(
    // this, nanogui::MessageDialog::Type::Warning,
    //"Invalid Selection",
    //"To select a channel you must select a select a region "
    //"using "
    //"the rectangle tool first.");
    //}
    //});
    // b->setTooltip("Select Channel");

    // b = new ToolButton(selectionButtons, ENTYPO_ICON_DATABASE);
    // b->setFlags(Button::NormalButton);
    // b->setCallback([this] {
    // if (imageView->rectSelectionValid()) {
    // selectMask();
    //}
    // else {
    // auto dlg = new nanogui::MessageDialog(
    // this, nanogui::MessageDialog::Type::Warning,
    //"Invalid Selection",
    //"To select a channel you must select a select a region "
    //"using "
    //"the rectangle tool first.");
    //}
    //});
    // b->setTooltip("Select Mask");

    // b = new ToolButton(selectionButtons, ENTYPO_ICON_BACK);
    // b->setFlags(Button::NormalButton);
    // b->setCallback([this] {
    // imageView->clearCurrentSelection();
    //[>        parametersControls->setVisible(false);<]
    //// rectVisibleCheckbox->setChecked(true);
    //[>rectVisibleCheckbox->callback()(true);<]
    // selectedChannelLabel->setCaption(
    //"  Selected Channel: None                 "
    //"                                         ");
    //});
    // b->setTooltip("Select Channel");

    //[>    initVisibilityWidgets();<]
    //[>initEditParmWidgets();<]

    /*    tabsWindow = new Window(controlsWindow, "");*/
    // tabsWindow->setLayout(new BoxLayout(nanogui::Orientation::Vertical,
    /*nanogui::Alignment::Fill, 0, 0));*/

    TabWidget *tabWidget = controlsWindow->add<TabWidget>();

    editsLayer = tabWidget->createTab("Layer Edits");
    editsLayer->setLayout((new BoxLayout(nanogui::Orientation::Vertical,
                                         nanogui::Alignment::Fill, 0, 0)));

    /*    selectedChannelLabel =*/
    // new Label(editsLayer,
    //"  Selected Channel: None                 "
    //"                                         ",
    //"sans-bold");

    Widget *channelsLayer = tabWidget->createTab("Selected Channels");
    channelsLayer->setLayout(new GroupLayout());
    VScrollPanel *vscroll = new VScrollPanel(channelsLayer);
    channelIconsImgPanel = new ImagePanel(vscroll);
    channelIconsImgPanel->setCallback([this](int index) {
        if (!mSelectedChannels[mChannelOrderingScore[index].second]) {
            imageView->setSelectedChannelId(
                mChannelOrderingScore[index].second);
            mSelectedChannels[mChannelOrderingScore[index].second] = true;
        }
        else {
            mSelectedChannels[mChannelOrderingScore[index].second] = false;
        }
    });

    tabWidget->setActiveTab(0);

    initVisibilityWidgets();
    initEditParmWidgets();
}

void EditTransferScreen::initMovablePanel()
{
    /*    moveablePanel = new Window(this, "");*/
    // moveablePanel->setLayout(new BoxLayout(nanogui::Orientation::Vertical,
    // nanogui::Alignment::Fill, 0, 0));

    // TabWidget *tabWidget = moveablePanel->add<TabWidget>();

    // Widget *editsLayer = tabWidget->createTab("Edits");
    // editsLayer->setLayout(new GroupLayout());

    // Widget *channelsLayer = tabWidget->createTab("Channels");
    /*channelsLayer->setLayout(new GroupLayout());*/
}

void EditTransferScreen::initVisibilityWidgets()
{

    cout << "initVisibilityWidgets " << endl;

    /*    GridLayout *layout = new GridLayout(nanogui::Orientation::Horizontal,
     * 2,*/
    // nanogui::Alignment::Fill, 5, 5);
    // layout->setColAlignment(
    //{nanogui::Alignment::Maximum, nanogui::Alignment::Fill});
    /*layout->setSpacing(0, 0);*/

    /*    Window *selectionPanelVisibilityButtons = new Window(editsLayer,*/
    //"");  // Visibility
    //// Options
    // selectionPanelVisibilityButtons->setLayout(new BoxLayout(
    /*nanogui::Orientation::Horizontal, nanogui::Alignment::Middle, 5, 5));*/

    /*    {*/
    /*        new
     * Label(selectionPanelVisibilityButtons,
     * "Create Mask:  ",*/
    //[>"sans-bold");<]

    /*        Window *maskTypeButtons
     * =*/
    //// new
    //// Window(selectionPanelVisibilityButtons,
    //// "");

    //// maskTypeButtons->setLayout(new
    //// BoxLayout(
    /*nanogui::Orientation::Horizontal,
     * nanogui::Alignment::Fill, 5,
     * 5));*/

    // Button *emptyMaskButton =
    // new Button(selectionPanelVisibilityButtons, "New Layer Edit");

    // emptyMaskButton->setTooltip("Create empty mask");
    // emptyMaskButton->setCallback(
    //[this]() { createMask(MaskInitType::Empty); });

    // Button *autoMaskButton =
    // new Button(selectionPanelVisibilityButtons, "Auto Mask");
    // autoMaskButton->setTooltip(
    //"Auto create mask based on "
    //"selection");
    // autoMaskButton->setCallback(
    //[this]() { createMask(MaskInitType::Auto); });
    /*}*/

    VScrollPanel *vscroll = new VScrollPanel(editsLayer);
    editIconsImgPanel = new ImagePanel(vscroll);
    editIconsImgPanel->setCallback([this](int index) {
        if (index == 0) {
            createMask(MaskInitType::Empty);
        }
        else {
            selectMask(index - 1);
        }
    });

    /*    new
     * Label(selectionPanelVisibilityButtons,
     * "Visibility Modes:  ",*/
    //"sans-bold");

    // Window *toggleVisibilityButtons =
    // new
    // Window(selectionPanelVisibilityButtons,
    // "");

    // toggleVisibilityButtons->setLayout(new
    // BoxLayout(
    // nanogui::Orientation::Horizontal,
    // nanogui::Alignment::Fill, 5, 5));

    //{
    // showCompositeToggleBtn = new
    // Button(toggleVisibilityButtons,
    // "B");
    // showCompositeToggleBtn->setTooltip("Blend
    // All Channels");
    // showCompositeToggleBtn->setPushed(true);
    // showCompositeToggleBtn->setChangeCallback([this](bool
    // state) {
    //// No callback needed??
    // if (state) {
    // imageView->mShowSelectedChannel =
    // !state;
    // imageView->mToggleShowMask =
    // !state;
    //}
    //});
    // showCompositeToggleBtn->setFlags(Button::RadioButton);
    //}

    //{
    // showChannelToggleBtn = new
    // Button(toggleVisibilityButtons,
    // "C");
    // showChannelToggleBtn->setTooltip("Selected
    // Channel Only");
    // showChannelToggleBtn->setFlags(Button::RadioButton);
    // showChannelToggleBtn->setChangeCallback([this](bool
    // state) {
    // if (imageView->channelSelected())
    // {
    // imageView->mShowSelectedChannel =
    // state;
    //}
    // else {
    // showChannelToggleBtn->setPushed(false);
    // showCompositeToggleBtn->setPushed(true);
    //}
    //});

    // showMaskToggleBtn = new
    // Button(toggleVisibilityButtons,
    // "M");
    // showMaskToggleBtn->setTooltip("Mask
    // Only");
    // showMaskToggleBtn->setFlags(Button::RadioButton);
    // showMaskToggleBtn->setChangeCallback([this](bool
    // state) {
    //[>            cout <<
    //"imageView->mCurrentSelectionIndex
    //"<]
    //[><<
    // imageView->mCurrentSelectionIndex
    //<< endl;<]
    // if
    // (imageView->validMaskSelected()) {
    // imageView->mToggleShowMask =
    // state;
    //}
    // else {
    // showMaskToggleBtn->setPushed(false);
    // showCompositeToggleBtn->setPushed(true);
    //}
    //});
    //}

    //{
    // new
    // Label(selectionPanelVisibilityButtons,
    //"Show Selection Rectangle Only:  ",
    //"sans-bold");

    // rectVisibleCheckbox = new
    // CheckBox(
    // selectionPanelVisibilityButtons,
    // "",
    //[this](bool state) {
    // imageView->mRectVisible = state;
    //});
    // rectVisibleCheckbox->setChecked(true);
    /*}*/
}

string to_string_2dp(float val)
{
    stringstream stream;
    stream << fixed << std::setprecision(2) << val;
    return stream.str();
}

string getGammaCaption(float val)
{
    return string("Gamma [" + to_string_2dp(val) + "]");
}

string getMinHistCaption(float val)
{
    return string("Hist Min [" + to_string_2dp(val) + "]");
}

string getMaxHistCaption(float val)
{
    return string("Hist Max [" + to_string_2dp(val) + "]");
}

string getMinHistOutputCaption(float val)
{
    return string("Output Min [" + to_string_2dp(val) + "]");
}

string getMaxHistOutputCaption(float val)
{
    return string("Output Max [" + to_string_2dp(val) + "]");
}

string getExposureCaption(float val)
{
    return string("Exposure [" + to_string_2dp(val) + "]");
}

string getHueCaption(float val)
{
    return string("Hue [" + to_string_2dp(val) + "]");
}

string getSaturationCaption(float val)
{
    return string("Saturation [" + to_string_2dp(val) + "]");
}

string getLightnessCaption(float val)
{
    return string("Lightness [" + to_string_2dp(val) + "]");
}

string getBrightnessCaption(float val)
{
    return string("Brightness [" + to_string_2dp(val) + "]");
}

string getContrastCaption(float val)
{
    return string("Contrast [" + to_string_2dp(val) + "]");
}

void EditTransferScreen::initEditParmWidgets()
{
    /*    cout << "initEditParmWidgets" << endl;*/

    parametersControls = controlsWindow->add<TabWidget>();

    GridLayout *layout = new GridLayout(nanogui::Orientation::Horizontal, 2,
                                        nanogui::Alignment::Minimum, 5, 5);
    /*    layout->setColAlignment(*/
    //{nanogui::Alignment::Minimum, nanogui::Alignment::Minimum});
    // layout->setRowAlignment(
    //{nanogui::Alignment::Minimum, nanogui::Alignment::Maximum});
    /*layout->setSpacing(0, 0);*/

    Widget *maskOptionsWidget = parametersControls->createTab("Mask Options");
    /*    Window *maskOptionButtons =*/
    // new Window(maskOptionButtons, "");  // Visibility Options
    maskOptionsWidget->setLayout(layout);
    /*maskOptionButtons->setWidth(300);*/

    {
        new Label(maskOptionsWidget, "Apply Mask:  ", "sans-bold");

        applyMaskCheckBox =
            new CheckBox(maskOptionsWidget, "Apply", [this](bool state) {
                mSelectedEdit->applyMaskToLayer(state);
                /*                mChannelEditsManager->applyMaskToLayer(*/
                // imageView->getSelectedChannelId(),
                /*imageView->getSelectedMaskId(), state);*/
            });
        applyMaskCheckBox->setChecked(false);
    }

    {
        new Label(maskOptionsWidget, "Auto Mask:  ", "sans-bold");

        Button *b = new Button(maskOptionsWidget, "Auto");
        b->setChangeCallback([this](bool state) {
            if (imageView->validMaskSelected() && state) {
                autoMaskRegion();
            }
        });
    }

    {
        new Label(maskOptionsWidget, "Invert Mask:  ", "sans-bold");

        Button *b = new Button(maskOptionsWidget, "Invert");
        b->setChangeCallback([this](bool state) {
            if (imageView->validMaskSelected() && state) {
                mSelectedEdit->invertMask();
                mChannelEditsManager->updateMasks(
                    mSelectedEdit->getChannelIndexes(), true);
            }
        });
    }

    {
        new Label(maskOptionsWidget, "Brush Tool:  ", "sans-bold");

        Button *b = new Button(maskOptionsWidget, "On");
        b->setFlags(Button::ToggleButton);
        b->setChangeCallback(
            [this](bool state) { imageView->setBrushState(state); });
    }

    {
        new Label(maskOptionsWidget, "Brush Size :", "sans-bold");
        auto intBox = new IntBox<int>(maskOptionsWidget);
        intBox->setEditable(true);
        intBox->setFixedSize(Vector2i(100, 30));
        intBox->setValue(10);
        intBox->setUnits("px");
        intBox->setDefaultValue("0");
        intBox->setFontSize(16);
        intBox->setFormat("[1-9][0-9]*");
        intBox->setSpinnable(true);
        intBox->setMinValue(1);
        intBox->setValueIncrement(1);
        intBox->setCallback(
            [this](int value) { imageView->setBrushSize(value); });
    }

    {
        new Label(maskOptionsWidget, "Brush Colour:  ", "sans-bold");
        Theme *theme = new Theme(nvgContext());
        theme->mWindowFillUnfocused = Color(255, 120, 0, 255);
        theme->mWindowFillFocused = Color(255, 120, 0, 255);
        theme->mWindowTitleUnfocused = Color(255, 120, 0, 255);
        theme->mWindowTitleFocused = Color(255, 120, 0, 255);
        theme->mWindowFillUnfocused = Color(255, 120, 0, 255);
        theme->mIconColor = Color(255, 120, 0, 255);

        auto textBox = new FloatBox<float>(maskOptionsWidget);
        textBox->setTheme(theme);
        textBox->setEditable(true);
        textBox->setFixedSize(Vector2i(100, 30));
        textBox->setValue(1.0);
        textBox->setUnits("i");
        textBox->setDefaultValue("1.0");
        textBox->setSpinnable(true);
        textBox->setFontSize(16);
        textBox->setValueIncrement(0.05);
        textBox->setFormat("0(.d+)?|1(.0+)?");
        textBox->setCallback(
            [this](float value) { imageView->setBrushValue(value); });
    }

    {
        new Label(maskOptionsWidget, "Blur Size: ", "sans-bold");
        auto intBox = new IntBox<int>(maskOptionsWidget);
        intBox->setEditable(true);
        intBox->setFixedSize(Vector2i(100, 30));
        intBox->setValue(0);
        intBox->setUnits("px");
        intBox->setDefaultValue("0");
        intBox->setFontSize(16);
        intBox->setFormat("[1-9][0-9]*");
        intBox->setSpinnable(true);
        intBox->setMinValue(0);
        intBox->setValueIncrement(1);
        intBox->setCallback(
            [this](int value) { mSelectedEdit->setBlurSize(value); });
    }

    {
        new Label(maskOptionsWidget, "Blur Sigma: ", "sans-bold");
        auto floatBox = new FloatBox<float>(maskOptionsWidget);
        floatBox->setEditable(true);
        floatBox->setFixedSize(Vector2i(100, 30));
        floatBox->setValue(2);
        floatBox->setUnits("px");
        floatBox->setDefaultValue("0");
        floatBox->setFontSize(16);
        floatBox->setFormat("[1-9][0-9]*");
        floatBox->setSpinnable(true);
        floatBox->setMinValue(0);
        floatBox->setValueIncrement(0.5);
        floatBox->setCallback(
            [this](float value) { mSelectedEdit->setBlurSigma(value); });
    }

    Widget *editParamsWidgetA =
        parametersControls->createTab("Edit Parameters A");

    // parametersControls = new Window(controlsWindow, "Edit Parameters");
    editParamsWidgetA->setLayout(new BoxLayout(
        nanogui::Orientation::Vertical, nanogui::Alignment::Fill, 10, 10));

    {
        new Label(editParamsWidgetA, "Removed:  ", "sans-bold");

        removeBtn = new Button(editParamsWidgetA, "Remove");
        removeBtn->setFlags(Button::ToggleButton);
        removeBtn->setChangeCallback([this](bool state) {
            float newMaxHistValue = 0.0;
            float newMinHistValue = 0.0;
            string newCaption = "Show";
            if (state) {
                mPrevHistMaxValue = maxHistRGBSlider->value();
                mPrevHistMinValue = minHistRGBSlider->value();
            }
            else {
                newMaxHistValue = mPrevHistMaxValue;
                newMinHistValue = mPrevHistMinValue;
                newCaption = "Remove";
            }

            /* mChannelEditsManager->setRemoveEdit(*/
            //// imageView->getSelectedChannelId(),
            /*imageView->getSelectedMaskId(), state);*/
            mSelectedEdit->setRemoveEdit(state);

            minOutRGBSlider->setValue(newMinHistValue);
            minOutRGBSlider->callback()(newMinHistValue);
            maxOutRGBSlider->setValue(newMaxHistValue);
            maxOutRGBSlider->callback()(newMaxHistValue);

            removeBtn->setCaption(newCaption);
        });
    }

    {
        Label *exposureLabel =
            new Label(editParamsWidgetA, getExposureCaption(0), "sans-bold");
        exposureSlider = new Slider(editParamsWidgetA);
        exposureSlider->setCallback([this, exposureLabel](float value) {
            mSelectedEdit->setExposure(value);
            exposureLabel->setCaption(
                getExposureCaption((value * 2.0 - 1.0) * 10.0));
        });
    }

    /*    graph = new Graph(editParamsWidget, "Local RBG Histogram");*/
    // VectorXf &func = graph->values();
    /*func.resize(256);*/

    Label *l = new Label(editParamsWidgetA, "Levels", "sans-bold");
    TabWidget *tabWidget = editParamsWidgetA->add<TabWidget>();

    {
        Widget *rgbLevelsWidget = tabWidget->createTab("RGB");
        rgbLevelsWidget->setLayout((new BoxLayout(
            nanogui::Orientation::Vertical, nanogui::Alignment::Fill, 0, 0)));

        Label *gammaLabel =
            new Label(rgbLevelsWidget, getGammaCaption(1), "sans-bold");
        gammaRGBSlider = new Slider(rgbLevelsWidget);
        gammaRGBSlider->setCallback([this, gammaLabel](float value) {
            auto val = mSelectedEdit->setRGBGamma(value);
            gammaLabel->setCaption(getGammaCaption(val));
        });

        Label *minHistLabel =
            new Label(rgbLevelsWidget, getMinHistCaption(0), "sans-bold");
        minHistRGBSlider = new Slider(rgbLevelsWidget);
        minHistRGBSlider->setCallback([this, minHistLabel](float value) {
            mSelectedEdit->setRGBHistMin(value);
            minHistLabel->setCaption(getMinHistCaption(value));
        });
        Label *maxHistLabel =
            new Label(rgbLevelsWidget, getMinHistCaption(1), "sans-bold");
        maxHistRGBSlider = new Slider(rgbLevelsWidget);
        maxHistRGBSlider->setCallback([this, maxHistLabel](float value) {

            mSelectedEdit->setRGBHistMax(value);
            maxHistLabel->setCaption(getMaxHistCaption(value));
        });

        Label *minOutLabel =
            new Label(rgbLevelsWidget, getMinHistOutputCaption(1), "sans-bold");
        minOutRGBSlider = new Slider(rgbLevelsWidget);
        minOutRGBSlider->setCallback([this, minOutLabel](float value) {

            mSelectedEdit->setRGBHistOutMin(value);
            minOutLabel->setCaption(getMinHistOutputCaption(value));
        });

        Label *maxOutLabel =
            new Label(rgbLevelsWidget, getMaxHistOutputCaption(1), "sans-bold");
        maxOutRGBSlider = new Slider(rgbLevelsWidget);
        maxOutRGBSlider->setCallback([this, maxOutLabel](float value) {

            mSelectedEdit->setRGBHistOutMax(value);
            maxOutLabel->setCaption(getMaxHistOutputCaption(value));
        });
    }

    //

    {
        Widget *rLevelsWidget = tabWidget->createTab("R");
        rLevelsWidget->setLayout((new BoxLayout(
            nanogui::Orientation::Vertical, nanogui::Alignment::Fill, 0, 0)));

        Label *gammaLabel =
            new Label(rLevelsWidget, getGammaCaption(1), "sans-bold");
        gammaRSlider = new Slider(rLevelsWidget);
        gammaRSlider->setCallback([this, gammaLabel](float value) {
            auto val = mSelectedEdit->setRGamma(value);
            gammaLabel->setCaption(getGammaCaption(val));
        });

        Label *minHistLabel =
            new Label(rLevelsWidget, getMinHistCaption(0), "sans-bold");
        minHistRSlider = new Slider(rLevelsWidget);
        minHistRSlider->setCallback([this, minHistLabel](float value) {
            mSelectedEdit->setRHistMin(value);
            minHistLabel->setCaption(getMinHistCaption(value));
        });

        Label *maxHistLabel =
            new Label(rLevelsWidget, getMinHistCaption(1), "sans-bold");
        maxHistRSlider = new Slider(rLevelsWidget);
        maxHistRSlider->setCallback([this, maxHistLabel](float value) {
            mSelectedEdit->setRHistMax(value);
            maxHistLabel->setCaption(getMaxHistCaption(value));
        });

        Label *minOutLabel =
            new Label(rLevelsWidget, getMinHistOutputCaption(1), "sans-bold");
        minOutRSlider = new Slider(rLevelsWidget);
        minOutRSlider->setCallback([this, minOutLabel](float value) {
            mSelectedEdit->setRHistOutMin(value);
            minOutLabel->setCaption(getMinHistOutputCaption(value));
        });

        Label *maxOutLabel =
            new Label(rLevelsWidget, getMaxHistOutputCaption(1), "sans-bold");
        maxOutRSlider = new Slider(rLevelsWidget);
        maxOutRSlider->setCallback([this, maxOutLabel](float value) {
            mSelectedEdit->setRHistOutMax(value);
            maxOutLabel->setCaption(getMaxHistOutputCaption(value));
        });
    }

    {
        Widget *gLevelsWidget = tabWidget->createTab("G");
        gLevelsWidget->setLayout((new BoxLayout(
            nanogui::Orientation::Vertical, nanogui::Alignment::Fill, 0, 0)));

        Label *gammaLabel =
            new Label(gLevelsWidget, getGammaCaption(1), "sans-bold");
        gammaGSlider = new Slider(gLevelsWidget);
        gammaGSlider->setCallback([this, gammaLabel](float value) {
            auto val = mSelectedEdit->setGGamma(value);
            gammaLabel->setCaption(getGammaCaption(val));
        });

        Label *minHistLabel =
            new Label(gLevelsWidget, getMinHistCaption(0), "sans-bold");
        minHistGSlider = new Slider(gLevelsWidget);
        minHistGSlider->setCallback([this, minHistLabel](float value) {
            mSelectedEdit->setGHistMin(value);
            minHistLabel->setCaption(getMinHistCaption(value));
        });
        Label *maxHistLabel =
            new Label(gLevelsWidget, getMinHistCaption(1), "sans-bold");
        maxHistGSlider = new Slider(gLevelsWidget);
        maxHistGSlider->setCallback([this, maxHistLabel](float value) {
            mSelectedEdit->setGHistMax(value);
            maxHistLabel->setCaption(getMaxHistCaption(value));
        });

        Label *minOutLabel =
            new Label(gLevelsWidget, getMinHistOutputCaption(1), "sans-bold");
        minOutGSlider = new Slider(gLevelsWidget);
        minOutGSlider->setCallback([this, minOutLabel](float value) {
            mSelectedEdit->setGHistOutMin(value);
            minOutLabel->setCaption(getMinHistOutputCaption(value));
        });

        Label *maxOutLabel =
            new Label(gLevelsWidget, getMaxHistOutputCaption(1), "sans-bold");
        maxOutGSlider = new Slider(gLevelsWidget);
        maxOutGSlider->setCallback([this, maxOutLabel](float value) {
            mSelectedEdit->setGHistOutMax(value);
            maxOutLabel->setCaption(getMaxHistOutputCaption(value));
        });
    }

    {
        Widget *bLevelsWidget = tabWidget->createTab("B");
        bLevelsWidget->setLayout((new BoxLayout(
            nanogui::Orientation::Vertical, nanogui::Alignment::Fill, 0, 0)));

        Label *gammaLabel =
            new Label(bLevelsWidget, getGammaCaption(1), "sans-bold");
        gammaBSlider = new Slider(bLevelsWidget);
        gammaBSlider->setCallback([this, gammaLabel](float value) {
            auto val = mSelectedEdit->setBGamma(value);
            gammaLabel->setCaption(getGammaCaption(val));
        });

        Label *minHistLabel =
            new Label(bLevelsWidget, getMinHistCaption(0), "sans-bold");
        minHistBSlider = new Slider(bLevelsWidget);
        minHistBSlider->setCallback([this, minHistLabel](float value) {
            mSelectedEdit->setBHistMin(value);
            minHistLabel->setCaption(getMinHistCaption(value));
        });

        Label *maxHistLabel =
            new Label(bLevelsWidget, getMinHistCaption(1), "sans-bold");
        maxHistBSlider = new Slider(bLevelsWidget);
        maxHistBSlider->setCallback([this, maxHistLabel](float value) {
            mSelectedEdit->setBHistMax(value);
            maxHistLabel->setCaption(getMaxHistCaption(value));
        });

        Label *minOutLabel =
            new Label(bLevelsWidget, getMinHistOutputCaption(1), "sans-bold");
        minOutBSlider = new Slider(bLevelsWidget);
        minOutBSlider->setCallback([this, minOutLabel](float value) {
            mSelectedEdit->setBHistOutMin(value);
            minOutLabel->setCaption(getMinHistOutputCaption(value));
        });

        Label *maxOutLabel =
            new Label(bLevelsWidget, getMaxHistOutputCaption(1), "sans-bold");
        maxOutBSlider = new Slider(bLevelsWidget);
        maxOutBSlider->setCallback([this, maxOutLabel](float value) {
            mSelectedEdit->setBHistOutMax(value);
            maxOutLabel->setCaption(getMaxHistOutputCaption(value));
        });
    }

    tabWidget->setActiveTab(0);

    Widget *editParamsWidgetB =
        parametersControls->createTab("Edit Parameters B");
    editParamsWidgetB->setLayout(new BoxLayout(
        nanogui::Orientation::Vertical, nanogui::Alignment::Fill, 10, 10));

    {
        Label *hueLabel =
            new Label(editParamsWidgetB, getHueCaption(0), "sans-bold");
        hueSlider = new Slider(editParamsWidgetB);
        hueSlider->setCallback([this, hueLabel](float value) {
            mSelectedEdit->setHue(value);
            hueLabel->setCaption(getHueCaption((value * 2.0 - 1.0) * 180.0));
        });

        Label *saturationLabel =
            new Label(editParamsWidgetB, getSaturationCaption(0), "sans-bold");
        saturationSlider = new Slider(editParamsWidgetB);
        saturationSlider->setCallback([this, saturationLabel](float value) {
            mSelectedEdit->setSaturation(value);
            saturationLabel->setCaption(
                getSaturationCaption((value * 2.0 - 1.0) * 100.0));
        });

        Label *lightnessLabel =
            new Label(editParamsWidgetB, getLightnessCaption(0), "sans-bold");
        lightnessSlider = new Slider(editParamsWidgetB);
        lightnessSlider->setCallback([this, lightnessLabel](float value) {
            mSelectedEdit->setLightness(value);
            lightnessLabel->setCaption(
                getLightnessCaption((value * 2.0 - 1.0) * 100.0));
        });
    }

    {
        Label *brightnessLabel =
            new Label(editParamsWidgetB, getBrightnessCaption(0), "sans-bold");
        brightnessSlider = new Slider(editParamsWidgetB);
        brightnessSlider->setCallback([this, brightnessLabel](float value) {
            mSelectedEdit->setBrightness(value);
            brightnessLabel->setCaption(
                getBrightnessCaption((value * 2.0 - 1.0) * 150.0));
        });

        Label *contrastLabel =
            new Label(editParamsWidgetB, getContrastCaption(0), "sans-bold");
        contrastSlider = new Slider(editParamsWidgetB);
        contrastSlider->setCallback([this, contrastLabel](float value) {
            mSelectedEdit->setContrast(value);
            contrastLabel->setCaption(
                getContrastCaption((value * 2.0 - 1.0) * 100.0));
        });
    }

    parametersControls->setActiveTab(0);
    /*    graph = new Graph(editParamsWidget, "Local RBG Histogram");*/
    // VectorXf &func = graph->values();
    /*func.resize(256);*/
}

void EditTransferScreen::updateGraph()
{

    /*    if (imageView->validMaskSelected()) {*/
    // int channelId = imageView->getSelectedChannelId();

    // cout << mFilenames[channelId + 1] << endl;
    // Mat localHist = mChannelEditsManager->getLocalHistogram(
    // channelId, imageView->getSelectedMaskId(),
    // mImageMats[channelId + 1]);

    // double min, max;
    // minMaxLoc(localHist, &min, &max);
    // double mid = (min + max) * 0.5;
    // float total = sum(localHist)[0];
    // VectorXf histValues(256);
    // for (auto i = 0; i < 256; i++) {
    // histValues(i) = localHist.at<float>(0, i) / float(mid);
    //// cout << histValues(i) << " ";
    //}
    //// cout << endl;
    // graph->setValues(histValues);

    // gammaSlider->setValue(0.5f);
    // minHistSlider->setValue(0);
    // maxHistSlider->setValue(1.0);
    /*}*/
}
void EditTransferScreen::selectChannel(Mat *roi_mask, vector<int> &ordering)
{

    cout << "EditTransferScreen::selectChannel" << endl;
    mChannelOrderingScore.clear();
    ordering.clear();
    float total_energy;
    mChannelSelector->selectUniqueChannel(roi_mask, mChannelOrderingScore,
                                          total_energy);
    cout << mChannelOrderingScore[0].second << endl;
    auto filename = mFilenames[mChannelOrderingScore[0].second + 1];
    cout << filename << endl;
    auto found = filename.find("/_.") + 1;
    /*    string labelText = "  Selected Channel: " + filename.substr(found);*/
    /*selectedChannelLabel->setCaption(labelText);*/

    // selectedLocalTochannelIcons = {0, 5, 7, 1, 2, 3, 4, 6, 8, 9};

    vector<pair<int, string>> reorderedImageIcons;
    vector<bool> selectedLocalTochannelIcons;
    float energy_so_far = 0.0;
    for (auto i : mChannelOrderingScore) {
        reorderedImageIcons.push_back(imageIcons[i.second]);
        cout << i.second << endl;
        ordering.push_back(i.second);
        if (energy_so_far < (0.6 * total_energy))
            mSelectedChannels[i.second] = true;
        else
            mSelectedChannels[i.second] = false;

        energy_so_far += i.first;
        selectedLocalTochannelIcons.push_back(mSelectedChannels[i.second]);
    }

    /*    selectedLocalTochannelIcons.clear();*/
    // selectedLocalTochannelIcons = {0, 5, 7, 1, 2, 3, 4, 6, 8, 9};
    // mSelectedChannels[0] = true;
    // mSelectedChannels[5] = true;
    /*mSelectedChannels[7] = true;*/

    channelIconsImgPanel->setImages(reorderedImageIcons);
    channelIconsImgPanel->setSelected(selectedLocalTochannelIcons);
    parametersControls->setVisible(false);

    vector<bool> selectedEdit(editIcons.size());
    selectedEdit[0] = true;

    editIconsImgPanel->setSelected(selectedEdit);

    resizeWidgets(false);

    /*    showChannelToggleBtn->setPushed(true);*/
    // imageView->mShowSelectedChannel = true;
    // showMaskToggleBtn->setPushed(false);
    // imageView->mToggleShowMask = false;
    /*showCompositeToggleBtn->setPushed(false);*/

    mSelectedMaterialId = 0;
    /*    float max_val = 0;*/
    // for (auto i = 0; i < mMaterialIdMats.size(); i++) {
    // Mat dst_roi = Mat(mMaterialIdMats[i]);
    // dst_roi.copyTo(dst_roi, *roi_mask);
    // Scalar m = mean(dst_roi);

    // if (m[0] > max_val) {
    // max_val = m[0];
    // mSelectedMaterialId = i;
    //}
    /*}*/
}

bool EditTransferScreen::keyboardEvent(int key, int /*scancode*/, int action,
                                       int modifiers)
{
    return imageView->keyboardEvent(key, 0, action, modifiers);
}

void EditTransferScreen::createMask(MaskInitType initType)
{

    if (!imageView->channelSelected()) {

        auto dlg = new nanogui::MessageDialog(
            this, nanogui::MessageDialog::Type::Warning, "No Channel Selected",
            "A channel must be selected to create a mask.");

        return;
    }

    cout << "createMask " << endl;
    auto channel_indexs = imageView->getSelectedChannelIds();

    Mat channel = mImageMats[channel_indexs[0] + 1];
    for (auto i = 1; i < channel_indexs.size(); i++) {
        channel += mImageMats[channel_indexs[i] + 1];
    }

    Mat initMask(channel.rows, channel.cols, CV_32FC3, Scalar(1, 1, 1));

    cout << "channel_indexs[0] " << channel_indexs[0] << " " << channel.rows
         << " " << channel.cols << endl;

    /*    auto rect = imageView->getRect();*/
    // if (initType == MaskInitType::Auto) {
    // Segmentor segmentor(channel, rect);
    // initMask = segmentor.getMask();
    /*}*/

    JWH::Rectangle rect(10, 10, 10, 10);

    MaskEdit mask_edit(initMask, rect, mFilenames[channel_indexs[0] + 1],
                       channel, Mat(), mSelectedMaterialId);

    cout << "Created mask_edit channel_indexs[0] " << channel_indexs[0] << endl;

    mask_edit.setChannelIndexes(channel_indexs);

    Mat mask = mask_edit.getMask();

    int selectedMaskId =
        mChannelEditsManager->push_back(channel_indexs, move(mask_edit));

    auto tempEdit =
        mChannelEditsManager->getEdit(channel_indexs[0], selectedMaskId);

    mMaskEdits.push_back(tempEdit);

    Mat rgbaMask(mask.rows, mask.cols, CV_8UC4);

    cvtColor(mask, rgbaMask, CV_RGB2RGBA, 4);
    rgbaMask.convertTo(rgbaMask, CV_8UC4, 255);

    int img_id = nvgCreateImageRGBA(mNVGContext, channel.rows, channel.cols, 0,
                                    rgbaMask.data);

    editIcons.push_back(
        make_pair(img_id, "mask" + to_string(editIcons.size())));

    editIconsImgPanel->setImages(editIcons);

    /*    vector<bool> selectedEdit;*/
    // for (auto i = 0; i < editIcons.size() - 1; i++)
    // selectedEdit.push_back(false);

    // selectedEdit.push_back(true);
    /*editIconsImgPanel->setSelected(selectedEdit);*/

    cout << "Select mask " << mMaskEdits.size() << endl;
    selectMask(mMaskEdits.size() - 1);
}

void EditTransferScreen::autoMaskRegion()
{
    if (!imageView->validMaskSelected()) {

        auto dlg = new nanogui::MessageDialog(
            this, nanogui::MessageDialog::Type::Warning, "No Mask Selected",
            "A mask must be selected to auto mask region.");

        return;
    }

    cout << "autoMaskRegionMask " << endl;
    auto channel_indexs = imageView->getSelectedChannelIds();

    Mat channel = mImageMats[channel_indexs[0] + 1];
    for (auto i = 1; i < channel_indexs.size(); i++) {
        channel += mImageMats[channel_indexs[i] + 1];
    }

    Mat initMask(channel.rows, channel.cols, CV_32FC3, Scalar(0, 0, 0));

    Mat polygon_mask = imageView->getPolygonMask();
    Segmentor segmentor(channel, polygon_mask);

    /*    Mat mask = cv::imread(*/
    //"/Users/JamesHennessey/Projects/edit_transfer_interface/scenes/wine/"
    //"edited_view1/edits/white_bottle_highlight_match/"
    //"FB.WhiteLight_Reflection_Mask_2.png",
    // 1);

    // imshow("read_mask", mask);
    // cout << mask.rows << " " << mask.cols << " " << mask.channels() << endl;
    // cout << channel.rows << " " << channel.cols << endl;

    //// cvtColor(mask, mask, CV_GRAY2BGR);
    // mask.convertTo(mask, CV_32FC3);
    // cout << mask.rows << " " << mask.cols << " " << mask.channels() << endl;
    /*initMask = mask;*/
    initMask = segmentor.getMask();

    mSelectedEdit->copySegmentationToMask(initMask, polygon_mask);
    mChannelEditsManager->updateMasks(channel_indexs);
}

void EditTransferScreen::showParamsWidget()
{
    // imageView->showMask();
    // initEditParmWidgets();

    /*showMaskToggleBtn->setPushed(true);*/
    // imageView->mToggleShowMask = true;
    // showChannelToggleBtn->setPushed(false);
    // imageView->mShowSelectedChannel = false;
    /*showCompositeToggleBtn->setPushed(false);*/
    /*    applyMaskCheckBox->setChecked(true);*/
    // mChannelEditsManager->applyMaskToLayer(imageView->getSelectedChannelId(),
    // imageView->getSelectedMaskId(),
    /*true);*/

    updateGraph();
    parametersControls->setVisible(true);
    resizeWidgets(false);
}

void setValueAndCallbackSlider(Slider *slider, float value)
{
    slider->setValue(value);
    slider->callback()(value);
}

void EditTransferScreen::selectMask(int index)
{

    cout << "Select Mask " << index << endl;

    mselectedEditIcon = index;

    cout << mMaskEdits.size() << endl;
    mSelectedEdit = mMaskEdits[index];

    imageView->setSelectedChannelId(mSelectedEdit->getChannelIndex());
    imageView->setSelectedMaskId(mSelectedEdit->getMaskIndex());

    showParamsWidget();
    // updateGraph();

    setValueAndCallbackSlider(gammaRGBSlider, mSelectedEdit->getRGBGamma());
    setValueAndCallbackSlider(minHistRGBSlider, mSelectedEdit->getRGBHistMin());
    setValueAndCallbackSlider(maxHistRGBSlider, mSelectedEdit->getRGBHistMax());
    setValueAndCallbackSlider(minOutRGBSlider,
                              mSelectedEdit->getRGBHistOutMin());
    setValueAndCallbackSlider(maxOutRGBSlider,
                              mSelectedEdit->getRGBHistOutMax());

    setValueAndCallbackSlider(gammaRSlider, mSelectedEdit->getRGamma());
    setValueAndCallbackSlider(minHistRSlider, mSelectedEdit->getRHistMin());
    setValueAndCallbackSlider(maxHistRSlider, mSelectedEdit->getRHistMax());
    setValueAndCallbackSlider(minOutRSlider, mSelectedEdit->getRHistOutMin());
    setValueAndCallbackSlider(maxOutRSlider, mSelectedEdit->getRHistOutMax());

    setValueAndCallbackSlider(gammaGSlider, mSelectedEdit->getGGamma());
    setValueAndCallbackSlider(minHistGSlider, mSelectedEdit->getGHistMin());
    setValueAndCallbackSlider(maxHistGSlider, mSelectedEdit->getGHistMax());
    setValueAndCallbackSlider(minOutGSlider, mSelectedEdit->getGHistOutMin());
    setValueAndCallbackSlider(maxOutGSlider, mSelectedEdit->getGHistOutMax());

    setValueAndCallbackSlider(gammaBSlider, mSelectedEdit->getBGamma());
    setValueAndCallbackSlider(minHistBSlider, mSelectedEdit->getBHistMin());
    setValueAndCallbackSlider(maxHistBSlider, mSelectedEdit->getBHistMax());
    setValueAndCallbackSlider(minOutBSlider, mSelectedEdit->getBHistOutMin());
    setValueAndCallbackSlider(maxOutBSlider, mSelectedEdit->getBHistOutMax());

    setValueAndCallbackSlider(hueSlider, mSelectedEdit->getHue());
    setValueAndCallbackSlider(saturationSlider, mSelectedEdit->getSaturation());
    setValueAndCallbackSlider(lightnessSlider, mSelectedEdit->getLightness());

    setValueAndCallbackSlider(brightnessSlider, mSelectedEdit->getBrightness());
    setValueAndCallbackSlider(contrastSlider, mSelectedEdit->getContrast());

    setValueAndCallbackSlider(exposureSlider, mSelectedEdit->getExposure());

    std::cout << "Exposure set " << std::endl;
    applyMaskCheckBox->setChecked(mSelectedEdit->applyMask());

    removeBtn->setPushed(mSelectedEdit->getRemoveEdit());

    vector<bool> selectedEdits(editIcons.size());

    cout << "editIcons.size() " << editIcons.size() << endl;
    cout << "index " << index << endl;

    selectedEdits[index + 1] = true;
    editIconsImgPanel->setSelected(selectedEdits);

    mSelectedChannels = vector<bool>(imageIcons.size());
    for (auto id : mSelectedEdit->getChannelIndexes()) {
        mSelectedChannels[id] = true;
    }

    channelIconsImgPanel->setSelected(mSelectedChannels);
    channelIconsImgPanel->setImages(imageIcons);

    /*    auto edit =*/
    // mChannelEditsManager->getEdit(maskSelected.first, maskSelected.second);

    // imageView->mRectSelection = edit->getRectangle();
    /*imageView->showMask();*/
}

void EditTransferScreen::updateMaskIcon(int selectedChannel, int maskIndex)
{
    Mat mask = mSelectedEdit->getMask();

    cv::Mat newSrc(mask.size(), CV_32FC4, Scalar(1,1,1,1));
    
    int from_to[] = {0, 0, 1, 1, 2, 2};
    cv::mixChannels(&mask, 1, &newSrc, 1, from_to, 3);

    newSrc.convertTo(newSrc, CV_8UC4, 255.0);

    Mat resizedMat;
    resize(newSrc, resizedMat, Size(1000, 1000));

    int img_id =
        nvgCreateImageRGBA(mNVGContext, 1000, 1000, 0, resizedMat.data);

    editIcons[mselectedEditIcon + 1] =
        make_pair(img_id, "mask" + to_string(mselectedEditIcon + 1));
    editIconsImgPanel->setImages(editIcons);
}
