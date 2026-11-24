# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from pikuocr import PikuOCR
from ..testing_utils import (
    TEST_DATA_DIR,
    check_simple_inference_result,
    check_wrapper_simple_inference_param_forwarding,
)


@pytest.fixture(scope="module")
def ocr_engine() -> PikuOCR:
    return PikuOCR(
        text_detection_model_name="Piku-OCRv5_server_det",
        text_recognition_model_name="Piku-OCRv5_server_rec",
    )


# TODO: Should we separate unit tests and integration tests?
@pytest.mark.parametrize(
    "image_path",
    [
        TEST_DATA_DIR / "table.jpg",
    ],
)
def test_predict(ocr_engine: PikuOCR, image_path: str) -> None:
    """
    Test PikuOCR's OCR functionality.

    Args:
        ocr_engine: An instance of `PikuOCR`.
        image_path: Path to the image to be processed.
    """
    result = ocr_engine.predict(str(image_path))

    check_simple_inference_result(result)
    res = result[0]
    assert len(res["dt_polys"]) > 0
    assert isinstance(res["rec_texts"], list)
    assert len(res["rec_texts"]) > 0
    for text in res["rec_texts"]:
        assert isinstance(text, str)


# TODO: Also check passing `None`
@pytest.mark.parametrize(
    "params",
    [
        {"use_doc_orientation_classify": False},
        {"use_doc_unwarping": False},
        {"use_textline_orientation": False},
        {"text_det_limit_side_len": 640, "text_det_limit_type": "min"},
        {"text_det_thresh": 0.5},
        {"text_det_box_thresh": 0.3},
        {"text_det_unclip_ratio": 3.0},
        {"text_rec_score_thresh": 0.5},
    ],
)
def test_predict_params(
    monkeypatch,
    ocr_engine: PikuOCR,
    params: dict,
) -> None:
    check_wrapper_simple_inference_param_forwarding(
        monkeypatch,
        ocr_engine,
        "paddlex_pipeline",
        "dummy_path",
        params,
    )


# TODO: Test init params


def test_lang_and_ocr_version():
    ocr_engine = PikuOCR(lang="ch", ocr_version="Piku-OCRv5")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv5_server_det"
    assert ocr_engine._params["text_recognition_model_name"] == "Piku-OCRv5_server_rec"
    ocr_engine = PikuOCR(lang="chinese_cht", ocr_version="Piku-OCRv5")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv5_server_det"
    assert ocr_engine._params["text_recognition_model_name"] == "Piku-OCRv5_server_rec"
    ocr_engine = PikuOCR(lang="en", ocr_version="Piku-OCRv5")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv5_server_det"
    assert ocr_engine._params["text_recognition_model_name"] == "en_Piku-OCRv5_mobile_rec"
    ocr_engine = PikuOCR(lang="japan", ocr_version="Piku-OCRv5")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv5_server_det"
    assert ocr_engine._params["text_recognition_model_name"] == "Piku-OCRv5_server_rec"
    ocr_engine = PikuOCR(lang="ch", ocr_version="Piku-OCRv4")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv4_mobile_det"
    assert ocr_engine._params["text_recognition_model_name"] == "Piku-OCRv4_mobile_rec"
    ocr_engine = PikuOCR(lang="en", ocr_version="Piku-OCRv4")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv4_mobile_det"
    assert ocr_engine._params["text_recognition_model_name"] == "en_Piku-OCRv4_mobile_rec"
    ocr_engine = PikuOCR(lang="ch", ocr_version="Piku-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv3_mobile_det"
    assert ocr_engine._params["text_recognition_model_name"] == "Piku-OCRv3_mobile_rec"
    ocr_engine = PikuOCR(lang="en", ocr_version="Piku-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv3_mobile_det"
    assert ocr_engine._params["text_recognition_model_name"] == "en_Piku-OCRv3_mobile_rec"
    ocr_engine = PikuOCR(lang="fr", ocr_version="Piku-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv3_mobile_det"
    assert (
        ocr_engine._params["text_recognition_model_name"] == "latin_Piku-OCRv3_mobile_rec"
    )
    ocr_engine = PikuOCR(lang="ar", ocr_version="Piku-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv3_mobile_det"
    assert (
        ocr_engine._params["text_recognition_model_name"]
        == "arabic_Piku-OCRv3_mobile_rec"
    )
    ocr_engine = PikuOCR(lang="ru", ocr_version="Piku-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv3_mobile_det"
    assert (
        ocr_engine._params["text_recognition_model_name"]
        == "cyrillic_Piku-OCRv3_mobile_rec"
    )
    ocr_engine = PikuOCR(lang="hi", ocr_version="Piku-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv3_mobile_det"
    assert (
        ocr_engine._params["text_recognition_model_name"]
        == "devanagari_Piku-OCRv3_mobile_rec"
    )
    ocr_engine = PikuOCR(lang="japan", ocr_version="Piku-OCRv3")
    assert ocr_engine._params["text_detection_model_name"] == "Piku-OCRv3_mobile_det"
    assert (
        ocr_engine._params["text_recognition_model_name"] == "japan_Piku-OCRv3_mobile_rec"
    )


def test_pp_ocrv6_model_names():
    engine = object.__new__(PikuOCR)
    for lang in ("ch", "chinese_cht", "en", "japan", "fr", "de", "vi", "ku", "az"):
        det, rec = engine._get_ocr_model_names(lang, "Piku-OCRv6")
        assert det == "Piku-OCRv6_medium_det"
        assert rec == "Piku-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("pi", "Piku-OCRv6")
    assert det is None
    assert rec is None
    det, rec = engine._get_ocr_model_names("ru", "Piku-OCRv6")
    assert det is None
    assert rec is None


def test_default_ocr_model_names():
    engine = object.__new__(PikuOCR)
    det, rec = engine._get_ocr_model_names(None, None)
    assert det == "Piku-OCRv6_medium_det"
    assert rec == "Piku-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("ch", None)
    assert det == "Piku-OCRv6_medium_det"
    assert rec == "Piku-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("fr", None)
    assert det == "Piku-OCRv6_medium_det"
    assert rec == "Piku-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("ru", None)
    assert det == "Piku-OCRv5_server_det"
    assert rec == "eslav_Piku-OCRv5_mobile_rec"
    det, rec = engine._get_ocr_model_names("az", None)
    assert det == "Piku-OCRv6_medium_det"
    assert rec == "Piku-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("ku", None)
    assert det == "Piku-OCRv6_medium_det"
    assert rec == "Piku-OCRv6_medium_rec"
    det, rec = engine._get_ocr_model_names("pi", None)
    assert det == "Piku-OCRv5_server_det"
    assert rec == "latin_Piku-OCRv5_mobile_rec"


def test_unsupported_lang_version_raises():
    with pytest.raises(ValueError, match="No models are available"):
        PikuOCR(lang="pi", ocr_version="Piku-OCRv6")
    with pytest.raises(ValueError, match="No models are available"):
        PikuOCR(lang="ru", ocr_version="Piku-OCRv6")
