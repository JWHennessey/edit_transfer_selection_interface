#version 400

/*
** Gamma correction and levels functions
** */

#define GammaCorrection(color, gamma) pow(color, vec3(1.0 / gamma))


#define LevelsControlInputRange(color, minInput, maxInput) \
    min(max(color - vec3(minInput), vec3(0.0)) /           \
            (vec3(maxInput) - vec3(minInput)),             \
        vec3(10))

#define LevelsControlInput(color, minInput, gamma, maxInput) \
    GammaCorrection(LevelsControlInputRange(color, minInput, maxInput), gamma)



#define LevelsControlOutputRange(color, minOutput, maxOutput) \
    mix(vec3(minOutput), vec3(maxOutput), color)
// 0 1 1 0 1
#define LevelsControl(color, minInput, gamma, maxInput, minOutput, maxOutput) \
    LevelsControlOutputRange(                                                 \
        LevelsControlInput(color, minInput, gamma, maxInput), minOutput,      \
        maxOutput)

const vec3 colorspace_LabWts_ = vec3(95.0456, 100.0, 108.8754);

const float MAX_PARAMS = 60;
const float MAX_MASK_PER_CHANNEL = 10;

const float R_GAMMA_INDEX = 0 / MAX_PARAMS;
const float R_INPUT_MIN_HIST_INDEX = 1 / MAX_PARAMS;
const float R_OUTPUT_MIN_HIST_INDEX = 2 / MAX_PARAMS;
const float R_INPUT_MAX_HIST_INDEX = 3 / MAX_PARAMS;
const float R_OUTPUT_MAX_HIST_INDEX = 4 / MAX_PARAMS;

const float G_GAMMA_INDEX = 5 / MAX_PARAMS;
const float G_INPUT_MIN_HIST_INDEX = 6 / MAX_PARAMS;
const float G_OUTPUT_MIN_HIST_INDEX = 7 / MAX_PARAMS;
const float G_INPUT_MAX_HIST_INDEX = 8 / MAX_PARAMS;
const float G_OUTPUT_MAX_HIST_INDEX = 9 / MAX_PARAMS;

const float B_GAMMA_INDEX = 10 / MAX_PARAMS;
const float B_INPUT_MIN_HIST_INDEX = 11 / MAX_PARAMS;
const float B_OUTPUT_MIN_HIST_INDEX = 12 / MAX_PARAMS;
const float B_INPUT_MAX_HIST_INDEX = 13 / MAX_PARAMS;
const float B_OUTPUT_MAX_HIST_INDEX = 14 / MAX_PARAMS;

const float EXPOSURE = 47 /  MAX_PARAMS;

const float RGB_GAMMA_INDEX = 48 / MAX_PARAMS;
const float RGB_INPUT_MIN_HIST_INDEX = 49 / MAX_PARAMS;
const float RGB_OUTPUT_MIN_HIST_INDEX = 50 / MAX_PARAMS;
const float RGB_INPUT_MAX_HIST_INDEX = 51 / MAX_PARAMS;
const float RGB_OUTPUT_MAX_HIST_INDEX = 52 / MAX_PARAMS;

const float HUE_INDEX = 53 / MAX_PARAMS;
const float SATURATION_INDEX = 54 / MAX_PARAMS;
const float LIGHTNESSS_INDEX = 55 / MAX_PARAMS;

const float BRIGHTNESS_INDEX = 56 / MAX_PARAMS;
const float CONTRAST_INDEX = 57 / MAX_PARAMS;

const float BLUR_SIZE_INDEX = 58 / MAX_PARAMS;
const float BLUR_SIGMA_INDEX = 59 / MAX_PARAMS;

const float APPLY_MASK_INDEX = 15 / MAX_PARAMS;

// const float MASK_TYPE_INDEX = 6 / MAX_PARAMS;

const float epsilon = 0.000001;
const int MAX_CHANNELS = XXX;

//uniform int noChannels;
uniform int showOriginalRGB;
uniform int showMask;
uniform int showPolygon;
uniform int currentSelectionIndex;
uniform int showSelectedChannel;
uniform vec4 selectedRectangle;
uniform int singleEditIndex;
uniform int selectedChannels[MAX_CHANNELS];

uniform sampler2D input_rgb;
uniform sampler2D polygon_select;
uniform sampler2D mask;
uniform sampler3D render_passes;
uniform sampler3D edit_masks[MAX_CHANNELS];
uniform sampler2DArray edit_params;

in vec2 uv;

layout(location = 0) out vec4 colour;
layout(location = 1) out vec4 edit_colour;
layout(location = 2) out vec4 edit_delta_colour;
layout(location = 3) out vec4 mask_intensity;
layout(location = 4) out vec4 beauty_edit;

struct EditParams {
    float rgbMinInput;
    float rgbGamma;
    float rgbMaxInput;
    float rgbMinOutput;
    float rgbMaxOutput;

    float rMinInput;
    float rGamma;
    float rMaxInput;
    float rMinOutput;
    float rMaxOutput;

    float gMinInput;
    float gGamma;
    float gMaxInput;
    float gMinOutput;
    float gMaxOutput;

    float bMinInput;
    float bGamma;
    float bMaxInput;
    float bMinOutput;
    float bMaxOutput;

    float hue;
    float saturation;
    float lightness;

    float brightness;
    float contrast;

    float exposure;

    int blurSize;
    float blurSigma;

    bool applyMask;
    //    float maskType;
    float maskValue;
};

float normpdf(float x, float sigma)
{
	return 0.39894*exp(-0.5*x*x/(sigma*sigma))/sigma;
}

vec3 rgb2xyz( vec3 c ) {
    vec3 tmp;
    tmp.x = ( c.r > 0.04045 ) ? pow( ( c.r + 0.055 ) / 1.055, 2.4 ) : c.r / 12.92;
    tmp.y = ( c.g > 0.04045 ) ? pow( ( c.g + 0.055 ) / 1.055, 2.4 ) : c.g / 12.92,
    tmp.z = ( c.b > 0.04045 ) ? pow( ( c.b + 0.055 ) / 1.055, 2.4 ) : c.b / 12.92;
    const mat3 mat = mat3(
		0.4124, 0.3576, 0.1805,
        0.2126, 0.7152, 0.0722,
        0.0193, 0.1192, 0.9505 
	);
    return 100.0 * (tmp * mat);
}

vec3 xyz2lab( vec3 c ) {
    vec3 n = c / vec3(95.047, 100, 108.883);
    vec3 v;
    v.x = ( n.x > 0.008856 ) ? pow( n.x, 1.0 / 3.0 ) : ( 7.787 * n.x ) + ( 16.0 / 116.0 );
    v.y = ( n.y > 0.008856 ) ? pow( n.y, 1.0 / 3.0 ) : ( 7.787 * n.y ) + ( 16.0 / 116.0 );
    v.z = ( n.z > 0.008856 ) ? pow( n.z, 1.0 / 3.0 ) : ( 7.787 * n.z ) + ( 16.0 / 116.0 );
    return vec3(( 116.0 * v.y ) - 16.0, 500.0 * ( v.x - v.y ), 200.0 * ( v.y - v.z ));
}

vec3 rgb2lab( vec3 c ) {
    vec3 lab = xyz2lab( rgb2xyz( c ) );
    return vec3( lab.x / 100.0, 0.5 + 0.5 * ( lab.y / 127.0 ), 0.5 + 0.5 * ( lab.z / 127.0 ));
}

vec3 lab2xyz( vec3 c ) {
    float fy = ( c.x + 16.0 ) / 116.0;
    float fx = c.y / 500.0 + fy;
    float fz = fy - c.z / 200.0;
    return vec3(
         95.047 * (( fx > 0.206897 ) ? fx * fx * fx : ( fx - 16.0 / 116.0 ) / 7.787),
        100.000 * (( fy > 0.206897 ) ? fy * fy * fy : ( fy - 16.0 / 116.0 ) / 7.787),
        108.883 * (( fz > 0.206897 ) ? fz * fz * fz : ( fz - 16.0 / 116.0 ) / 7.787)
    );
}

vec3 xyz2rgb( vec3 c ) {
	const mat3 mat = mat3(
        3.2406, -1.5372, -0.4986,
        -0.9689, 1.8758, 0.0415,
        0.0557, -0.2040, 1.0570
	);
    vec3 v = (c / 100.0) * mat;
    vec3 r;
    r.x = ( v.r > 0.0031308 ) ? (( 1.055 * pow( v.r, ( 1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.r;
    r.y = ( v.g > 0.0031308 ) ? (( 1.055 * pow( v.g, ( 1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.g;
    r.z = ( v.b > 0.0031308 ) ? (( 1.055 * pow( v.b, ( 1.0 / 2.4 ))) - 0.055 ) : 12.92 * v.b;
    return r;
}

vec3 lab2rgb( vec3 c ) {
    return xyz2rgb( lab2xyz( vec3(100.0 * c.x, 2.0 * 127.0 * (c.y - 0.5), 2.0 * 127.0 * (c.z - 0.5)) ) );
}

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}


float getEditParamFloat(float param_u, int param_y, int z)
{
    float param_v = float(param_y) / MAX_MASK_PER_CHANNEL;
    vec4 colour = texture(edit_params, vec3(param_u, param_v, z));
    return colour.r;
}


vec3 brightnessContrast(vec3 rgb, float brightness, float contrast)
{
	vec3 colorContrasted = ((rgb - 0.5) * max(contrast, 0)) + 0.5;
	vec3 bright = colorContrasted + vec3(brightness,brightness,brightness);
  return bright;
}

vec3 czm_saturation(vec3 rgb, float adjustment)
{
    // Algorithm from Chapter 16 of OpenGL Shading Language
    const vec3 W = vec3(0.2125, 0.7154, 0.0721);
    vec3 intensity = vec3(dot(rgb, W));
    return mix(intensity, rgb, adjustment);
}


vec3 hueSaturationLightness(vec3 rgb, float hueAdjustment, float saturationAdjustment, float lightnessAdjustment)
{
/*    const mat3 toYIQ = mat3(0.299,     0.587,     0.114,*/
                            /*0.595716, -0.274453, -0.321263,*/
                            /*0.211456, -0.522591,  0.311135);*/
    
    /*const mat3 toRGB = mat3(1.0,  0.9563,  0.6210,*/
                            /*1.0, -0.2721, -0.6474,*/
                            /*1.0, -1.107,   1.7046);*/
    
    /*vec3 yiq = toYIQ * rgb;*/
    /*float hue = atan(yiq.z, yiq.y) + adjustment;*/
    /*float chroma = sqrt(yiq.z * yiq.z + yiq.y * yiq.y);*/
    
    /*vec3 color = vec3(yiq.x, chroma * cos(hue), chroma * sin(hue));*/
    /*return toRGB * color;*/

  vec3 hsv = rgb2hsv(rgb);
  hsv.x += hueAdjustment;
  hsv.y += saturationAdjustment;
  hsv.z += lightnessAdjustment;
  rgb = hsv2rgb(hsv);
  return rgb;
}

bool getEditParamBool(float param_u, int param_y, int z)
{
    float value = getEditParamFloat(param_u, param_y, z);
    return (value == 1.0);
}

EditParams getEditParams(int param_y, int z)
{
    EditParams params;
    params.rgbMinInput = getEditParamFloat(RGB_INPUT_MIN_HIST_INDEX, param_y, z);
    params.rgbGamma = getEditParamFloat(RGB_GAMMA_INDEX, param_y, z);
    params.rgbMaxInput = getEditParamFloat(RGB_INPUT_MAX_HIST_INDEX, param_y, z);
    params.rgbMinOutput = getEditParamFloat(RGB_OUTPUT_MIN_HIST_INDEX, param_y, z);
    params.rgbMaxOutput = getEditParamFloat(RGB_OUTPUT_MAX_HIST_INDEX, param_y, z);

    params.rMinInput = getEditParamFloat(R_INPUT_MIN_HIST_INDEX, param_y, z);
    params.rGamma = getEditParamFloat(R_GAMMA_INDEX, param_y, z);
    params.rMaxInput = getEditParamFloat(R_INPUT_MAX_HIST_INDEX, param_y, z);
    params.rMinOutput = getEditParamFloat(R_OUTPUT_MIN_HIST_INDEX, param_y, z);
    params.rMaxOutput = getEditParamFloat(R_OUTPUT_MAX_HIST_INDEX, param_y, z);

    params.gMinInput = getEditParamFloat(G_INPUT_MIN_HIST_INDEX, param_y, z);
    params.gGamma = getEditParamFloat(G_GAMMA_INDEX, param_y, z);
    params.gMaxInput = getEditParamFloat(G_INPUT_MAX_HIST_INDEX, param_y, z);
    params.gMinOutput = getEditParamFloat(G_OUTPUT_MIN_HIST_INDEX, param_y, z);
    params.gMaxOutput = getEditParamFloat(G_OUTPUT_MAX_HIST_INDEX, param_y, z);

    params.bMinInput = getEditParamFloat(B_INPUT_MIN_HIST_INDEX, param_y, z);
    params.bGamma = getEditParamFloat(B_GAMMA_INDEX, param_y, z);
    params.bMaxInput = getEditParamFloat(B_INPUT_MAX_HIST_INDEX, param_y, z);
    params.bMinOutput = getEditParamFloat(B_OUTPUT_MIN_HIST_INDEX, param_y, z);
    params.bMaxOutput = getEditParamFloat(B_OUTPUT_MAX_HIST_INDEX, param_y, z);
    
    params.hue = getEditParamFloat(HUE_INDEX, param_y, z);
    params.saturation = getEditParamFloat(SATURATION_INDEX, param_y, z);
    params.lightness = getEditParamFloat(LIGHTNESSS_INDEX, param_y, z);
   
    params.brightness = getEditParamFloat(BRIGHTNESS_INDEX, param_y, z);
    params.contrast = getEditParamFloat(CONTRAST_INDEX, param_y, z);

    params.exposure = getEditParamFloat(EXPOSURE, param_y, z);
   
    params.blurSize = int(getEditParamFloat(BLUR_SIZE_INDEX, param_y, z));
    params.blurSigma = getEditParamFloat(BLUR_SIGMA_INDEX, param_y, z);

    params.applyMask = getEditParamBool(APPLY_MASK_INDEX, param_y, z);
    // params.maskType = getEditParamFloat(MASK_TYPE_INDEX, param_y, z);
    return params;
}

bool isInRectangle()
{
    if (uv[0] < selectedRectangle[0]) return false;
    if (uv[1] < selectedRectangle[1]) return false;
    if (uv[0] > (selectedRectangle[0] + selectedRectangle[2])) return false;
    if (uv[1] > (selectedRectangle[1] + selectedRectangle[3])) return false;
    return true;
}

bool isChannelSelected() { return (currentSelectionIndex > -1); }
bool isSelectedShown() { return (showSelectedChannel == 1); }
bool isMaskShown() { return (showMask == 1); }

vec4 getTextureColour(int i, ivec3 size)
{

    vec4 edit_colour =
        texture(render_passes, vec3(uv[0], uv[1], float(i) / float(size.z)));
    return edit_colour;
}

bool hasEditMask(int id)
{
    ivec3 mask_size = textureSize(edit_masks[id], 0);
    ivec3 passes_size = textureSize(render_passes, 0);
    return (mask_size.x == passes_size.x && mask_size.y == passes_size.y);
}

bool debuggingTests(ivec3 channels_size, ivec3 params_size)
{

    if (params_size[1] != MAX_MASK_PER_CHANNEL) {
        colour = vec4(uv[0], 0.5, 0.5, 1);
        return true;
    }

    if (params_size[0] != MAX_PARAMS ||
        params_size[1] != MAX_MASK_PER_CHANNEL) {
        colour = vec4(uv[0], uv[1], 0.5, 1);
        return true;
    }

    if (params_size.z != channels_size.z) {
        colour = vec4(1, uv[1], 0, 1);
        return true;
    }
    return false;
}

// Return only the greyscale value
float getMaskValue(int c_id, int m_id, ivec3 size)
{
    float d = float(m_id) / float(size.z);
    vec4 mask_color = texture(edit_masks[c_id], vec3(uv[0], uv[1], d));
    return mask_color.r;
}

void debugAdjustmentLayer(vec3 input_colour, vec3 adjusted_colour)
{
    vec3 diff = adjusted_colour - input_colour;
    colour.b += 1.0 * length(diff);
}



float Gaussian (float sigma, float x)
{
    return exp(-(x*x) / (2.0 * sigma*sigma));
}

vec3 BlurredPixel (vec2 uv, int c_id, EditParams params, ivec3 channels_size)
{
    int halfSize = params.blurSize / 2;
    float total = 0.0;
    vec3 ret = vec3(0);
    ivec3 mask_size = textureSize(edit_masks[c_id], 0);
    for (int iy = 0; iy < params.blurSize; ++iy)
    {
        float fy = Gaussian (params.blurSigma, float(iy) - float(halfSize));
        float offsety = float(iy-halfSize) * (1.0 / float(mask_size.y));
        for (int ix = 0; ix < params.blurSize; ++ix)
        {
            float fx = Gaussian (params.blurSigma, float(ix) - float(halfSize));
            float offsetx = float(ix-halfSize) * (1.0 / float(mask_size.x));
            total += fx * fy;            
            ret += texture(render_passes, vec3(uv[0] + offsetx, uv[1] + offsety, float(c_id) / float(channels_size.z))).rgb * fx*fy;
        }
    }
    return ret / total;
}

vec3 applyEdit(vec3 inputColour, EditParams editParams)
{

    vec3 originalColour = inputColour;
    inputColour = inputColour * pow(2, editParams.exposure);

    //inputColour = czm_saturation(inputColour, editParams.saturation);
    inputColour = hueSaturationLightness(inputColour, editParams.hue, editParams.saturation, editParams.lightness);

    inputColour = brightnessContrast(inputColour, editParams.brightness, editParams.contrast);
   
    vec3 tempRGB = LevelsControl(
        vec3(inputColour.r, inputColour.g, inputColour.b), editParams.rgbMinInput, editParams.rgbGamma,
        editParams.rgbMaxInput, editParams.rgbMinOutput, editParams.rgbMaxOutput);


    vec3 tempR = LevelsControl(
        vec3(tempRGB.r, tempRGB.r, tempRGB.r), editParams.rMinInput, editParams.rGamma,
        editParams.rMaxInput, editParams.rMinOutput, editParams.rMaxOutput);

    vec3 tempG = LevelsControl(
        vec3(tempRGB.g, tempRGB.g, tempRGB.g), editParams.gMinInput, editParams.gGamma,
        editParams.gMaxInput, editParams.gMinOutput, editParams.gMaxOutput);

    vec3 tempB = LevelsControl(
        vec3(tempRGB.b, tempRGB.b, tempRGB.b), editParams.bMinInput, editParams.bGamma,
        editParams.bMaxInput, editParams.bMinOutput, editParams.bMaxOutput);

    vec3 adjusted_colour = vec3(tempR.r, tempG.g, tempB.b);

   // vec3 adjusted_colour_exposure = adjusted_colour;
   // adjusted_colour_exposure = adjusted_colour_exposure * pow(2, editParams.exposure);

    vec3 outputColour = adjusted_colour;//lab2rgb(adjusted_colour);// 

    
     



    outputColour = outputColour * (editParams.maskValue);
    /*outputColour.r = min(outputColour.r,1);*/
    /*outputColour.g = min(outputColour.g,1);*/
    /*outputColour.b = min(outputColour.b,1);*/
/*    if(tempL.x > inputLabColour.x)*/
    /*{*/
      /*outputColour.r += 0.7;*/
    /*}*/
    return outputColour;
}

vec3 applyEditsToTexture(int c_id, ivec3 channels_size)
{

    vec3 channel_edit_colour = getTextureColour(c_id, channels_size).rgb;

    // If there is no mask add channel texture to output colour
    if (!hasEditMask(c_id)) {
        return channel_edit_colour;
    }
    vec3 colourAfterEdit = vec3(0, 0, 0);
    ivec3 mask_size = textureSize(edit_masks[c_id], 0);

    if (singleEditIndex > -1) {
        EditParams editParams = getEditParams(singleEditIndex, c_id);
        editParams.maskValue = getMaskValue(c_id, singleEditIndex, mask_size);
        
        if(editParams.blurSize > 0 && editParams.maskValue > 0.01)
        {
           channel_edit_colour = BlurredPixel(uv, c_id, editParams, channels_size);
        }
        
        if (editParams.maskValue > 0.01) {
            colourAfterEdit = applyEdit(channel_edit_colour, editParams);
        }
        else {
            colourAfterEdit = channel_edit_colour;
        }
        return colourAfterEdit;
    }

    bool hasHadEditApplied = false;

    for (int m_id = 0; m_id < mask_size.z; m_id++) {
        EditParams editParams = getEditParams(m_id, c_id);
        editParams.maskValue = getMaskValue(c_id, m_id, mask_size);

        if (hasHadEditApplied) {
            channel_edit_colour = colourAfterEdit;
        }

        // Edit Should be appled
        if (editParams.applyMask && editParams.maskValue > 0.01) {

            if(editParams.blurSize > 0 && !hasHadEditApplied)
            {
               channel_edit_colour = BlurredPixel(uv, c_id, editParams, channels_size);
            }

            vec3 c = applyEdit(channel_edit_colour, editParams);
            vec3 delta = c - channel_edit_colour;
            if (length(delta) > epsilon || hasHadEditApplied) {
                if (!hasHadEditApplied) {
                    colourAfterEdit += c;
                    hasHadEditApplied = true;
                }
                else {
                    // Add only the delta if this is the second edit for a
                    // fragment
                    colourAfterEdit += delta;
                }
            }
        }
    }

    if (!hasHadEditApplied) {
        colourAfterEdit = channel_edit_colour;
    }

    return colourAfterEdit;
}

void main()
{
    colour = vec4(0, 0, 0, 0);
    edit_colour = vec4(0, 0, 0, 0);
    edit_delta_colour = vec4(0, 0, 0, 0);

    // Display original rgb image
    if (showOriginalRGB == 0) {
        colour = texture(input_rgb, uv);
        // colour.g = 0.7;
        edit_colour = colour;
        return;
    }

    ivec3 channels_size = textureSize(render_passes, 0);
    ivec3 params_size = textureSize(edit_params, 0);

    bool hasBug =
        debuggingTests(channels_size,
                       params_size);  // Make sure textures are the correct size

    if (hasBug) return;

    colour = vec4(0, 0, 0, 1);
    bool inRect = isInRectangle();
    bool channelSelected = isChannelSelected();
    bool isSelectedVisible = isSelectedShown();
    bool isMaskVisible = isMaskShown();
    // Composite and apply edits
    if (  // channelSelected && isSelectedVisible && !inRect && !isMaskVisible
        //! inRect ||
        (!isSelectedVisible && !isMaskVisible)) {  // && singleEditIndex == -1
        // Loop over all channels compositing them together
        for (int c_id = 0; c_id < channels_size.z; c_id++) {
            colour.xyz += applyEditsToTexture(c_id, channels_size);
        }
        colour.a = 1;
    }
    // Show only the mask
    else if (isMaskVisible) {
        colour = texture(mask, uv);
        // colour.r += 0.5;
    }
    // Show only single channel
    else {
        for (int i = 0; i < channels_size.z; i++) {
            if (selectedChannels[i] == 1) {
                colour.xyz += applyEditsToTexture(
                    i, channels_size);  // currentSelectionIndex,
                // colour.r += 0.2;
            }
            colour.a = 1;
        }
    }

    // Write out to frame buffers
    if (singleEditIndex > -1) {
        vec4 input_colour =
            getTextureColour(currentSelectionIndex, channels_size);
        edit_delta_colour = abs(input_colour - colour);
        edit_colour = colour;
        ivec3 mask_size = textureSize(edit_masks[currentSelectionIndex], 0);
        float mv =
            getMaskValue(currentSelectionIndex, singleEditIndex, mask_size);
        mask_intensity = vec4(mv, mv, mv, 1);
    }
    else {
        edit_colour = colour;
    }

    beauty_edit = edit_colour;

    float gamma = 1.6;
    colour.rgb = pow(colour.rgb, vec3(1.0 / gamma));
    colour.a = 1;

    if(showPolygon == 1)
    {
      colour += texture(polygon_select, uv);
      beauty_edit += texture(polygon_select, uv);
    }


}
