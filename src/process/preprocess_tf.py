import tensorflow as tf
import config

_PAD = float(getattr(config, "LETTERBOX_PAD_VALUE", 114.0 / 255.0))

@tf.function
def letterbox_tf(img, new_size=config.IMGSZ, pad_value=_PAD, scaleup=True):
    """
    img: float32 [H,W,3] in [0,1]
    return:
      img_lb: float32 [new_size,new_size,3]
      meta: float32 [5] = [orig_h, orig_w, scale, pad_x, pad_y]  (pad_x/pad_y in pixels of new_size space)
    """
    img = tf.convert_to_tensor(img, tf.float32)
    shape = tf.shape(img)
    orig_h = tf.cast(shape[0], tf.float32)
    orig_w = tf.cast(shape[1], tf.float32)

    new_size_f = tf.cast(new_size, tf.float32)

    # scale to fit in new_size (keep aspect)
    scale = new_size_f / tf.maximum(orig_h, orig_w)
    if not scaleup:
        scale = tf.minimum(scale, 1.0)

    new_h = tf.cast(tf.round(orig_h * scale), tf.int32)
    new_w = tf.cast(tf.round(orig_w * scale), tf.int32)
    new_h = tf.maximum(new_h, 1)
    new_w = tf.maximum(new_w, 1)

    img_rs = tf.image.resize(
        img, [new_h, new_w],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True
    )

    pad_h = tf.cast(new_size, tf.int32) - new_h
    pad_w = tf.cast(new_size, tf.int32) - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img_lb = tf.pad(
        img_rs,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        constant_values=tf.cast(pad_value, tf.float32)
    )
    img_lb = tf.ensure_shape(img_lb, [new_size, new_size, 3])

    meta = tf.stack([
        orig_h,
        orig_w,
        tf.cast(scale, tf.float32),
        tf.cast(pad_left, tf.float32),
        tf.cast(pad_top, tf.float32),
    ], axis=0)

    return img_lb, meta


@tf.function
def decode_and_letterbox(img_path, new_size=config.IMGSZ, pad_value=_PAD, scaleup=True):
    """
    img_path: tf.string scalar
    return img_lb, meta
    """
    img_path = tf.ensure_shape(img_path, [])
    img_bytes = tf.io.read_file(img_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return letterbox_tf(img, new_size=new_size, pad_value=pad_value, scaleup=scaleup)



