import streamlit as st
import cv2
import numpy as np
import joblib
import os
from PIL import Image
import io


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Image Orientation Corrector",
    page_icon="🔄",
    layout="wide"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_PATH = "orientation_model.pkl"

IMAGE_SIZE = (128, 128)

ANGLES = [0, 90, 180, 270]


# ============================================================
# HOG FEATURE EXTRACTOR
# MUST BE SAME AS TRAINING
# ============================================================

hog = cv2.HOGDescriptor(
    _winSize=(128, 128),
    _blockSize=(32, 32),
    _blockStride=(16, 16),
    _cellSize=(16, 16),
    _nbins=9
)


def extract_features(image):

    # Convert RGB to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Resize exactly like training
    gray = cv2.resize(
        gray,
        IMAGE_SIZE
    )

    # Extract HOG
    features = hog.compute(gray)

    # Flatten
    features = features.flatten()

    return features


# ============================================================
# LOAD SVM MODEL
# ============================================================

@st.cache_resource
def load_model():

    saved_data = joblib.load(MODEL_PATH)

    model = saved_data["model"]

    image_size = saved_data["image_size"]

    return model, image_size


# ============================================================
# SESSION STATE
# ============================================================

if "verified" not in st.session_state:
    st.session_state.verified = False

if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = False

if "corrected_image" not in st.session_state:
    st.session_state.corrected_image = None

if "last_file" not in st.session_state:
    st.session_state.last_file = None

if "selected_angle" not in st.session_state:
    st.session_state.selected_angle = None


# ============================================================
# HEADER
# ============================================================

st.title("🔄 Image Orientation Corrector")

st.write(
    "Automatically detect the orientation of an image, "
    "verify the result, correct it using OpenCV, "
    "and download the corrected image."
)

st.divider()


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ orientation_model.pkl was not found."
    )

    st.write(
        "Make sure orientation_model.pkl is in the same "
        "folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, saved_image_size = load_model()

except Exception as e:

    st.error(
        "❌ Could not load orientation_model.pkl"
    )

    st.write(str(e))

    st.stop()


# ============================================================
# UPLOAD
# ============================================================

st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# WAIT FOR IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload an image to begin."
    )

    st.stop()


# ============================================================
# RESET FOR NEW IMAGE
# ============================================================

if st.session_state.last_file != uploaded_file.name:

    st.session_state.verified = False

    st.session_state.manual_mode = False

    st.session_state.corrected_image = None

    st.session_state.selected_angle = None

    st.session_state.last_file = uploaded_file.name


# ============================================================
# READ IMAGE
# ============================================================

try:

    image_pil = Image.open(
        uploaded_file
    ).convert("RGB")

    image = np.array(
        image_pil
    )

except Exception:

    st.error(
        "❌ Could not read the uploaded image."
    )

    st.stop()


# ============================================================
# IMAGE INFORMATION
# ============================================================

height, width = image.shape[:2]

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "Width",
        f"{width}px"
    )

with info2:

    st.metric(
        "Height",
        f"{height}px"
    )

with info3:

    st.metric(
        "Model",
        "HOG + SVM"
    )


st.divider()


# ============================================================
# ORIGINAL IMAGE
# ============================================================

st.subheader("🖼️ Uploaded Image")

st.image(
    image_pil,
    use_container_width=True
)


st.divider()


# ============================================================
# AUTOMATIC DETECTION
# ============================================================

st.subheader("🧠 Automatic Orientation Detection")


try:

    # Extract exactly the same HOG features
    features = extract_features(image)

    # SVM prediction
    predicted_angle = model.predict(
        [features]
    )[0]

    predicted_angle = int(
        predicted_angle
    )


    # SVM probabilities
    probabilities = model.predict_proba(
        [features]
    )[0]


    # Find probability belonging to predicted class
    class_list = list(
        model.classes_
    )

    predicted_index = class_list.index(
        predicted_angle
    )

    confidence = (
        probabilities[predicted_index] * 100
    )


except Exception as e:

    st.error(
        "❌ Orientation detection failed."
    )

    st.write(str(e))

    st.stop()


# ============================================================
# DESCRIPTIONS
# ============================================================

orientation_description = {

    0:
        "The image is upright.",

    90:
        "The image is rotated 90° clockwise.",

    180:
        "The image is upside down.",

    270:
        "The image is rotated 270° clockwise."
}


correction_description = {

    0:
        "No rotation is required.",

    90:
        "Rotate 90° counter-clockwise.",

    180:
        "Rotate 180°.",

    270:
        "Rotate 90° clockwise."
}


# ============================================================
# DETECTION RESULT
# ============================================================

result1, result2 = st.columns(2)


with result1:

    st.metric(
        "Detected Orientation",
        f"{predicted_angle}°"
    )

    st.write(
        orientation_description[
            predicted_angle
        ]
    )


with result2:

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.write(
        "Required correction:"
    )

    st.info(
        correction_description[
            predicted_angle
        ]
    )


st.divider()


# ============================================================
# VERIFY
# ============================================================

st.subheader("✅ Verify Detection")

st.write(
    "Look at the uploaded image and verify whether "
    "the detected orientation is correct."
)


# ============================================================
# VERIFY BUTTONS
# ============================================================

if (
    not st.session_state.verified
    and not st.session_state.manual_mode
):

    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "✅ Yes, Detection Is Correct",
            use_container_width=True,
            type="primary"
        ):

            st.session_state.verified = True

            st.session_state.selected_angle = (
                predicted_angle
            )

            st.rerun()


    with col2:

        if st.button(
            "❌ No, Choose Correct Orientation",
            use_container_width=True
        ):

            st.session_state.manual_mode = True

            st.rerun()


# ============================================================
# MANUAL CORRECTION SELECTION
# ============================================================

if (
    st.session_state.manual_mode
    and not st.session_state.verified
):

    st.warning(
        "The automatic prediction is incorrect. "
        "Select the actual orientation."
    )


    manual_options = [
        "0° — Upright",
        "90° — Rotated clockwise",
        "180° — Upside down",
        "270° — Rotated clockwise"
    ]


    manual_selection = st.radio(
        "Actual orientation:",
        manual_options
    )


    manual_angle = int(
        manual_selection.split("°")[0]
    )


    if st.button(
        "✅ Confirm Orientation",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.verified = True

        st.session_state.selected_angle = (
            manual_angle
        )

        st.session_state.manual_mode = False

        st.rerun()


# ============================================================
# CORRECTION
# ============================================================

if st.session_state.verified:

    final_angle = (
        st.session_state.selected_angle
    )


    st.success(
        f"✅ Orientation verified: {final_angle}°"
    )


    st.write(
        "Correction required: "
        + correction_description[
            final_angle
        ]
    )


    st.divider()


    st.subheader("🔄 Correct Image")


    if st.session_state.corrected_image is None:

        if st.button(
            "🔄 CORRECT IMAGE",
            use_container_width=True,
            type="primary"
        ):


            # =================================================
            # 0 DEGREE
            # =================================================

            if final_angle == 0:

                corrected_image = image.copy()


            # =================================================
            # 90 DEGREE CLOCKWISE
            # → COUNTER-CLOCKWISE
            # =================================================

            elif final_angle == 90:

                corrected_image = cv2.rotate(
                    image,
                    cv2.ROTATE_90_COUNTERCLOCKWISE
                )


            # =================================================
            # 180 DEGREE
            # =================================================

            elif final_angle == 180:

                corrected_image = cv2.rotate(
                    image,
                    cv2.ROTATE_180
                )


            # =================================================
            # 270 DEGREE CLOCKWISE
            # → CLOCKWISE 90
            # =================================================

            elif final_angle == 270:

                corrected_image = cv2.rotate(
                    image,
                    cv2.ROTATE_90_CLOCKWISE
                )


            # Save result
            st.session_state.corrected_image = (
                corrected_image
            )

            st.rerun()


# ============================================================
# BEFORE / AFTER
# ============================================================

if st.session_state.corrected_image is not None:

    st.divider()

    st.subheader("✨ Correction Result")

    st.write(
        "Before and after comparison:"
    )


    before, after = st.columns(2)


    with before:

        st.write("### Before")

        st.image(
            image,
            use_container_width=True
        )


    with after:

        st.write("### After")

        st.image(
            st.session_state.corrected_image,
            use_container_width=True
        )


    st.success(
        "🎉 Image successfully corrected!"
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    corrected_pil = Image.fromarray(
        st.session_state.corrected_image
    )


    buffer = io.BytesIO()


    corrected_pil.save(
        buffer,
        format="PNG"
    )


    st.download_button(
        label="⬇️ DOWNLOAD CORRECTED IMAGE",
        data=buffer.getvalue(),
        file_name="corrected_image.png",
        mime="image/png",
        use_container_width=True
    )


    st.divider()


    # ========================================================
    # RESET
    # ========================================================

    if st.button(
        "🔄 Upload Another Image",
        use_container_width=True
    ):

        st.session_state.verified = False

        st.session_state.manual_mode = False

        st.session_state.corrected_image = None

        st.session_state.selected_angle = None

        st.session_state.last_file = None

        st.rerun()