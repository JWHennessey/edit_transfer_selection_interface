#include "ChannelEditsManager.h"

ChannelEditsManager::ChannelEditsManager()
    : mAllEditParameters(
          make_pair(GLTexture2DArray("EditParamsTextureArray"), nullptr))
{
}

void ChannelEditsManager::init(int size)
{
    mAllEdits.clear();
    mAll3DTextures.clear();

    mAllEdits = vector<vector<shared_ptr<MaskEdit>>>(size);
    for (auto i = 0; i < size; i++) {
        GL3DTexture texture3d("Mask3DTexture" + to_string(i));
        mAll3DTextures.emplace_back(std::move(texture3d), nullptr);
    }

    mAllEditParameters.second = mAllEditParameters.first.init(size);
}

int ChannelEditsManager::push_back(vector<int> channelIds, MaskEdit edit)
{
    cout << "ChannelEditsManager::push_back " << channelIds[0] << endl;
    cout << "channelIds.size() " << channelIds.size() << endl;
    shared_ptr<MaskEdit> editPtr = make_shared<MaskEdit>(move(edit));
    editPtr->setChannelIndexes(channelIds);

    editPtr->setCallback([this] { updateEditParametersTexture(); });

    for (auto i : channelIds) {
        cout << "push_back " << i << endl;
        editPtr->setChannelIndex(i);
        editPtr->setMaskIndex(mAllEdits[i].size());
        mAllEdits[i].push_back(editPtr);
        mAll3DTextures[i].second = mAll3DTextures[i].first.load(mAllEdits[i]);
        updateEditParametersTexture(i);
        cout << mAllEdits[i].size() << endl;
    }
    return mAllEdits[channelIds[0]].size() - 1;
}

void ChannelEditsManager::updateMasks(vector<int> channelIds, bool all_masks)
{
    for (auto channelId : channelIds) {
       mAll3DTextures[channelId].first.update(mAllEdits[channelId], all_masks);
    }
}

shared_ptr<MaskEdit> ChannelEditsManager::getEdit(int channelId, int maskId)
{
    return mAllEdits[channelId][maskId];
}

ImageDataType& ChannelEditsManager::getMaskGLData(int channelId, int maskId)
{
    /*    assert(mAllEdits.size() > channelId);*/
    /*assert(mAllEdits[channelId].size() > maskId);*/
    return mAllEdits[channelId][maskId]->maskGLData();
}

bool ChannelEditsManager::channelHasMask(int channelId)
{
    return (mAllEdits.size() > channelId && mAllEdits[channelId].size() > 0);
}

Image3DDataType& ChannelEditsManager::getMask3DTexture(int channelId)
{
    return mAll3DTextures[channelId];
}

ImageData2DArray& ChannelEditsManager::getEditParameters()
{
    return mAllEditParameters;
}

void ChannelEditsManager::updateEditParametersTexture(int channelId)
{
    for (auto i = 0; i < mAllEdits.size(); i++) {
        if (mAllEdits[i].size() > 0) {
            mAllEditParameters.first.updateTexture(
                i, mAllEdits[i], mAllEditParameters.second.get());
        }
    }
}

pair<int, int> ChannelEditsManager::selectMask(JWH::Rectangle rect)
{
    auto channelMaskIds = make_pair(-1, -1);
    float maxArea = 0.0;
    for (auto i = 0; i < mAllEdits.size(); i++) {
        for (auto j = 0; j < mAllEdits[i].size(); j++) {
            float area = mAllEdits[i][j]->intersectionArea(rect);
            if ((channelMaskIds.first == -1 && area > 0.0) || maxArea < area) {
                maxArea = area;
                channelMaskIds.first = i;
                channelMaskIds.second = j;
            }
        }
    }
    return channelMaskIds;
}

int ChannelEditsManager::noChannelMasks(int channelId)
{
    return mAllEdits[channelId].size();
}

/*Mat ChannelEditsManager::getLocalHistogram(int channelId, int maskId,*/
// Mat channel)
//{
// Mat mask = mAllEdits[channelId][maskId]->getMask();

// channel.convertTo(channel, CV_8UC3, 255);
// cvtColor(channel, channel, COLOR_BGR2GRAY);

// mask.convertTo(mask, CV_8UC3, 255);
// cvtColor(mask, mask, COLOR_BGR2GRAY);

// Mat hist;
// calcHist(&channel, 1, 0, mask, hist, 1, &histSize, &histRange, uniform,
// accumulate_hist);

// return hist;
//}

// void ChannelEditsManager::invertMask(int channelId, int maskId)
//{
// assert(mAllEdits.size() > channelId);
// assert(mAllEdits[channelId].size() > maskId);
// mAllEdits[channelId][maskId]->invertMask();
// mAll3DTextures[channelId].second =
// mAll3DTextures[channelId].first.load(mAllEdits[channelId]);
//}

// float ChannelEditsManager::setGammaValue(int channelId, int maskId, float
// gamma)
//{

// JWH::convertToGamma<float>(&gamma);
//// cout << gamma << endl;

// mAllEdits[channelId][maskId]->setGamma(gamma);
// updateEditParametersTexture(channelId);
// return gamma;
//}

// void ChannelEditsManager::setHistMinValue(int channelId, int maskId,
// float value)
//{
// mAllEdits[channelId][maskId]->setHistMin(value);
// updateEditParametersTexture(channelId);
//}
// void ChannelEditsManager::setHistMaxValue(int channelId, int maskId,
// float value)
//{
// mAllEdits[channelId][maskId]->setHistMax(value);
// updateEditParametersTexture(channelId);
//}

// float ChannelEditsManager::getGammaValue(int channelId, int maskId)
//{
// float gamma = mAllEdits[channelId][maskId]->getGamma();

// JWH::convertFromGamma<float>(&gamma);

// return gamma;
//}

// float ChannelEditsManager::getHistMinValue(int channelId, int maskId)
//{
// return mAllEdits[channelId][maskId]->getHistMin();
//}

// float ChannelEditsManager::getHistMaxValue(int channelId, int maskId)
//{
// return mAllEdits[channelId][maskId]->getHistMax();
//}

// void ChannelEditsManager::applyMaskToLayer(int channelId, int maskId,
// bool state)
//{
// mAllEdits[channelId][maskId]->applyMaskToLayer(state);
// updateEditParametersTexture(channelId);
//}

// bool ChannelEditsManager::applyMask(int channelId, int maskId)
//{
// return mAllEdits[channelId][maskId]->applyMask();
//}

// bool ChannelEditsManager::getRemoveEdit(int channelId, int maskId)
//{
// return mAllEdits[channelId][maskId]->getRemoveEdit();
//}
// void ChannelEditsManager::setRemoveEdit(int channelId, int maskId, bool
// value)
//{
// mAllEdits[channelId][maskId]->setRemoveEdit(value);
//}

/*void ChannelEditsManager::setMaskType(int channelId, int maskId, float
 * value)*/
//{
// mAllEdits[channelId][maskId].setMaskType(value);
// updateEditParametersTexture(channelId);
/*}*/
