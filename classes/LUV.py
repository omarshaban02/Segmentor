import numpy as np


def luv_mapping(img):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    shape = r.shape

    b = (b / 255).flatten()
    g = (g / 255).flatten()
    r = (r / 255).flatten()

    def gamma_correction(value):
        value = np.asarray(value)
        condition = value <= 0.04045
        res = np.where(
            condition,
            value / 12.92,
            ((value + 0.055) / 1.055) ** 2.4
        )
        return res

    r = gamma_correction(r)
    g = gamma_correction(g)
    b = gamma_correction(b)
    rgb = np.array([r, g, b])

    converting_mat = [[0.412453, 0.357580, 0.180423],
                      [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]]

    x, y, z = np.matmul(converting_mat, rgb)

    # Calculate the chromatic coordinates
    u_dash = 4 * x / (x + 15 * y + 3 * z)
    v_dash = 9 * y / (x + 15 * y + 3 * z)

    un = 0.19793943
    vn = 0.46831096

    # Calculate L
    y_gt_idx = np.argwhere(y > 0.008856)
    y_le_idx = np.argwhere(y <= 0.008856)

    l = np.zeros_like(y)
    l[y_gt_idx] = (116 * y[y_gt_idx] ** (1 / 3)) - 16
    l[y_le_idx] = 903.3 * y[y_le_idx]

    # Calculate u and v
    u = 13 * l * (u_dash - un)
    v = 13 * l * (v_dash - vn)

    # Conversion to 8-bit
    l = 255 / 100 * l
    u = 225 / 354 * (u + 134)
    v = 255 / 262 * (v + 140)

    # Reshape after flattening
    l = l.reshape(shape)
    u = u.reshape(shape)
    v = v.reshape(shape)

    luv = np.array([l, u, v], np.int64).T
    luv = np.fliplr(np.rot90(luv, 3))
    return luv
