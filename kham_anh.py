import cv2
import os
import numpy as np

def make_mosaic(img, tile_size, tile_dir):
  """
  Tạo một khảm cắt dán từ một ảnh.

  Args:
    img: Ảnh đầu vào.
    tile_size: Kích thước của mỗi mảnh ghép.
    tile_dir: Thư mục chứa các mảnh ghép.

  Returns:
    Khảm cắt dán.
  """

  # Tạo ra một thư viện các mảnh ghép.
  tiles = []
  for filename in os.listdir(tile_dir):
    tile = cv2.imread(os.path.join(tile_dir, filename))
    tile = cv2.resize(tile, dsize=(tile_size, tile_size))
    tiles.append(tile)

  # Lặp qua mỗi mảnh ghép trong thư viện.
  mosaic = np.zeros_like(img)
  for i in range(0, img.shape[1], tile_size):
    for j in range(0, img.shape[0], tile_size):
      # Tìm vị trí của mảnh ghép trong ảnh đầu vào.
      min_diff = np.inf
      best_tile = None
      for tile in tiles:
        patch = img[j:j+tile_size, i:i+tile_size]
        if patch.shape == tile.shape:
          diff = cv2.sumElems((patch - tile)**2)
          diff = sum(diff)
          if diff < min_diff:
            min_diff = diff
            best_tile = tile

      # Đặt mảnh ghép vào vị trí đó trong kết quả.
      if best_tile is not None:
        mosaic[j:j+tile_size, i:i+tile_size] = best_tile

  return mosaic

# Đọc ảnh đầu vào.
img = cv2.imread('tiles/hai_dang.jpg')

# Tạo khảm cắt dán.
mosaic = make_mosaic(img, 100, 'tiles')

# Hiển thị khảm cắt dán.
cv2.imshow('Mosaic', mosaic)
cv2.waitKey(0)