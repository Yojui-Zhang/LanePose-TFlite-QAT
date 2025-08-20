

import tensorflow as tf

'''
預測 Teacher 模型結果進行比較
'''

def normalize_teacher_pred(y, expected_C):
    """
    將 teacher 輸出標準化為 [B, N, C]。
    - 若輸入已是 [B, N, C]（最後維度 == expected_C），直接回傳。
    - 若輸入是 [B, C, N]，嘗試 transpose -> [B, N, C] 並檢查最後維度是否為 expected_C。
    - 否則在 graph 中會觸發 assert（會中止），並給出有意義錯誤訊息。
    注意：此實作完全使用 TF ops，可在 @tf.function 中呼叫。
    """
    y = tf.convert_to_tensor(y)
    # 確保 rank==3（會在 graph 中觸發檢查）
    tf.debugging.assert_rank(y, 3, message="normalize_teacher_pred: input rank must be 3 (B,?,?)")

    # 動態 shape
    sh = tf.shape(y)
    dim1 = sh[1]
    dim2 = sh[2]
    expected = tf.cast(expected_C, dtype=dim2.dtype)

    # 判斷最後維度是否等於 expected_C
    cond_last_is_expected = tf.equal(dim2, expected)

    def _return_as_is():
        return y

    def _try_transpose_and_check():
        y_t = tf.transpose(y, perm=[0, 2, 1])  # [B, N, C] <- [B, C, N]
        # 檢查 transpose 後最後維度是 expected_C，若不是會觸發 assert
        tf.debugging.assert_equal(tf.shape(y_t)[2], expected,
                                  message=("normalize_teacher_pred: after transpose, "
                                           "last dim != expected_C"))
        return y_t

    result = tf.cond(cond_last_is_expected, _return_as_is, _try_transpose_and_check)
    return result