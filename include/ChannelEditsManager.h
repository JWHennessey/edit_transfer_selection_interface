#ifndef CHANNEL_EDITS_MANAGER_H
#define CHANNEL_EDITS_MANAGER_H

#include "ChannelSelector.h"
#include "MaskEdit.h"

using namespace std;

class ChannelEditsManager {
   public:
    ChannelEditsManager();
    void init(int size);
    int push_back(vector<int> channelIds, MaskEdit edit);
    shared_ptr<MaskEdit> getEdit(int channelId, int maskId);

    ImageDataType& getMaskGLData(int channelId, int maskId);
    bool channelHasMask(int channelId);

    Image3DDataType& getMask3DTexture(int channelId);
    ImageData2DArray& getEditParameters();

    pair<int, int> selectMask(JWH::Rectangle rect);

    inline int noChannels() { return mAll3DTextures.size(); }

    void updateEditParametersTexture(int channelId = 0);
    int noChannelMasks(int channelId);
    void updateMasks(vector<int> channelIds, bool all_masks = false);

    /*    Mat getLocalHistogram(int channelId, int maskId, Mat channel);*/
    // void invertMask(int channelId, int maskId);

    // float setGammaValue(int channelId, int maskId, float gamma);
    // void setHistMinValue(int channelId, int maskId, float value);
    // void setHistMaxValue(int channelId, int maskId, float value);

    // float getGammaValue(int channelId, int maskId);
    // float getHistMinValue(int channelId, int maskId);
    // float getHistMaxValue(int channelId, int maskId);

    // void applyMaskToLayer(int channelId, int maskId, bool state);

    // bool applyMask(int channelId, int maskId);
    // bool getRemoveEdit(int channelId, int maskId);
    // void setRemoveEdit(int channelId, int maskId, bool value);

    // void setMaskType(int channelId, int maskId, float value);

   private:
    vector<vector<shared_ptr<MaskEdit>>> mAllEdits;
    Image3DDataTypeVec mAll3DTextures;
    ImageData2DArray mAllEditParameters;
    // Fucntions
};

#endif
