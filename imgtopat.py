import numpy as np
import tensorflow as tf
import cv2
from tensorflow.contrib.learn.python.learn.estimators import kmeans as kmeans_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.training import input as input_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

import vgg

VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
CONTENT_LAYER = 'relu4_2'
GRID_PX = 5
LEARNING_RATE = 1e1
ITERATIONS = 1000
ITER_PER_CHECKPOINT = 10
# CVT_TYPE = cv2.COLOR_BGR2YUV
# INV_CVT_TYPE = cv2.COLOR_YUV2BGR
CVT_TYPE = cv2.COLOR_BGR2LAB
INV_CVT_TYPE = cv2.COLOR_LAB2BGR
# CVT_TYPE = cv2.COLOR_BGR2LUV
# INV_CVT_TYPE = cv2.COLOR_LUV2BGR
# CVT_TYPE = cv2.COLOR_BGR2YCR_CB
# INV_CVT_TYPE = cv2.COLOR_YCR_CB2BGR

MAX_SIDE_PX = 800
MAX_AREA = MAX_SIDE_PX * MAX_SIDE_PX

RGB_TO_YUV = np.array([[0.299, 0.587, 0.114],
                       [-0.14713, -0.28886, 0.436],
                       [0.615, -0.51499, -0.10001]], dtype=np.float32)

YUV_TO_RGB = np.array([[1., 0., 1.13983],
                       [1., -0.39465, -0.58060],
                       [1., 2.03211, 0.]], dtype=np.float32)


def imread(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    # bgr to rgb
    img = img[..., ::-1]

    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    # rgb to bgr
    img = img[..., ::-1]
    cv2.imwrite(path, img)


def generate_pattern(content):
    # crop an image at center so new h and w are multiple of GRID_PX
    x_px = content.shape[0]
    y_px = content.shape[1]
    x_px_lost = x_px % GRID_PX
    y_px_lost = y_px % GRID_PX
    start_x_px_idx = x_px_lost // 2
    start_y_px_idx = y_px_lost // 2

    content = content[
              start_x_px_idx: start_x_px_idx + x_px // GRID_PX * GRID_PX,
              start_y_px_idx: start_y_px_idx + y_px // GRID_PX * GRID_PX,
              :
              ]
    imsave('./output/cropped.png', content)

    # k means

    def input_fn(use_mini_batch = False,
                 mini_batch_steps_per_iteration = 1,
                 batch_size=None,
                 points=None,
                 randomize=None,
                 num_epochs=None):
        """Returns an input_fn that randomly selects batches from given points."""
        num_points = points.shape[0]
        if randomize is None:
            randomize = (use_mini_batch and
                         mini_batch_steps_per_iteration <= 1)

        def _fn():
            x = constant_op.constant(points)
            if batch_size == num_points:
                return input_lib.limit_epochs(x, num_epochs=num_epochs), None
            if randomize:
                indices = random_ops.random_uniform(
                    constant_op.constant([batch_size]),
                    minval=0,
                    maxval=num_points - 1,
                    dtype=dtypes.int32,
                    seed=10)
            else:
                # We need to cycle through the indices sequentially. We create a queue
                # to maintain the list of indices.
                q = data_flow_ops.FIFOQueue(num_points, dtypes.int32, ())

                # Conditionally initialize the Queue.
                def _init_q():
                    with ops.control_dependencies(
                            [q.enqueue_many(math_ops.range(num_points))]):
                        return control_flow_ops.no_op()

                init_q = control_flow_ops.cond(q.size() <= 0, _init_q,
                                               control_flow_ops.no_op)
                with ops.control_dependencies([init_q]):
                    offsets = q.dequeue_many(batch_size)
                    with ops.control_dependencies([q.enqueue_many(offsets)]):
                        indices = array_ops.identity(offsets)
            batch = array_ops.gather(x, indices)
            return (input_lib.limit_epochs(batch, num_epochs=num_epochs), None)

        return _fn

    num_clusters = 64
    use_mini_batch = False
    mini_batch_steps_per_iteration = 1
    batch_size = 1000
    relative_tolerance = 1e-4
    # relative_tolerance = None

    kmeans = kmeans_lib.KMeansClustering(
        num_clusters=num_clusters,
        initial_clusters=kmeans_lib.KMeansClustering.KMEANS_PLUS_PLUS_INIT,
        distance_metric=kmeans_lib.KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE,
        use_mini_batch=use_mini_batch,
        mini_batch_steps_per_iteration=mini_batch_steps_per_iteration,
        relative_tolerance=relative_tolerance)

    points = content.reshape([-1,3])

    kmeans.fit(
        input_fn=input_fn(
            use_mini_batch=use_mini_batch,
            mini_batch_steps_per_iteration=mini_batch_steps_per_iteration,
            batch_size=batch_size,
            points=points,
            randomize=None,
            num_epochs=1
        ),
        # Force it to train until the relative tolerance monitor stops it.
        steps=None)

    palette = kmeans.clusters()

    p = palette.reshape((num_clusters, 1, 3))
    palette_img = np.concatenate((p, p, p, p, p, p, p, p, p, p, p, p, p, p), axis=1)
    imsave('./output/palette.jpg', palette_img)





    #convert color scheme
    content_bgr = content[..., ::-1] / 255.
    content_cvt = cv2.cvtColor(content_bgr, CVT_TYPE)

    content_features = {}

    image_shape = content.shape
    x_px = image_shape[0]
    y_px = image_shape[1]
    batch_shape = (1,) + image_shape
    image_area = x_px * y_px
    if image_area > MAX_AREA:
        sf = image_area / MAX_AREA


    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=batch_shape)
        net, mean_pixel = vgg.net(VGG_PATH, image)

        content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
            feed_dict={image: np.array([vgg.preprocess(content, mean_pixel)])})

    with tf.Graph().as_default():

        x_g_px = x_px // GRID_PX + (0 if x_px % GRID_PX == 0 else 1)
        y_g_px = y_px // GRID_PX + (0 if y_px % GRID_PX == 0 else 1)

        # fill initialization colors array
        init_colors = np.zeros((x_g_px, y_g_px, 3), dtype=np.float32)
        for i in range(x_g_px):
            for j in range(y_g_px):
                x_s_px = i * GRID_PX
                x_e_px = min((i + 1) * GRID_PX, x_px)
                y_s_px = j * GRID_PX
                y_e_px = min((j + 1) * GRID_PX, y_px)

                xs_cvt = content_cvt[x_s_px: x_e_px, y_s_px: y_e_px, :]
                xs_init_cvt_color = np.mean(xs_cvt, axis=(0, 1))
                xs_init_bgr_color = cv2.cvtColor(xs_init_cvt_color[np.newaxis, np.newaxis, :], INV_CVT_TYPE)
                xs_init_rgb_color = xs_init_bgr_color[..., ::-1] * 255.
                xs_init = vgg.preprocess(xs_init_rgb_color, mean_pixel)
                init_colors[i, j] = xs_init[0, 0]

        indices = np.zeros((x_px, y_px), dtype=np.int32)
        colors = []
        t_c = 0

        def add_new_color(t_c, color):
            colors.append(color)
            return t_c + 1

        for i in range(x_px):
            for j in range(y_px):
                i_g = i // GRID_PX
                j_g = j // GRID_PX
                i_shift = i % GRID_PX
                j_shift = j % GRID_PX
                # get initial color
                color = init_colors[i_g, j_g, :].reshape((3))
                if i_shift == 0 and j_shift == 0:
                    # create color variable
                    indices[i][j] = t_c
                    t_c = add_new_color(t_c, color)
                else:
                    # copy color
                    indices[i][j] = indices[i_g * GRID_PX][j_g * GRID_PX]

        colors_init = np.stack(colors)
        colors_v = tf.Variable(colors_init)
        flat_indices = tf.constant(indices.reshape((-1, 1)))
        flat_image = tf.gather_nd(colors_v, flat_indices)
        image = tf.reshape(flat_image, shape=batch_shape)

        net, _ = vgg.net(VGG_PATH, image)

        # content loss
        content_loss = tf.nn.l2_loss(net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_features[
            CONTENT_LAYER].size

        # overall loss
        loss = content_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        # optimization
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            init_img = sess.run(image).reshape(image_shape)
            init_img = vgg.unprocess(init_img, mean_pixel)
            imsave('./output/init.png'.format(i), init_img)
            init_img_bgr = init_img[..., ::-1].astype(np.float32) / 255.
            init_img_cvt = cv2.cvtColor(init_img_bgr, CVT_TYPE)
            _, c2, c3 = cv2.split(init_img_cvt)

            for i in range(ITERATIONS):
                loss_content = sess.run([content_loss])
                print('iteration {} perceptual loss: {}'.format(i, loss_content))
                sess.run(train_step)

                if i % ITER_PER_CHECKPOINT == 0:
                    result = image.eval()
                    result = result.reshape(image_shape)
                    result = vgg.unprocess(result, mean_pixel)
                    imsave('./output/result_{}.png'.format(i), result)

                    result_bgr = result[..., ::-1].astype(np.float32) / 255.

                    # rgb to bgr
                    result_cvt = cv2.cvtColor(result_bgr, CVT_TYPE)
                    c1, _, _ = cv2.split(result_cvt)
                    merged = cv2.merge((c1, c2, c3))
                    result = cv2.cvtColor(merged, INV_CVT_TYPE) * 255.
                    # bgr to rgb
                    result = result[..., ::-1]
                    imsave('./output/result_{}_color.png'.format(i), result)


def main():
    content = imread('./input/input.jpg')
    generate_pattern(content)


if __name__ == '__main__':
    main()
