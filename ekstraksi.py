from PIL import Image

def gambar_ke_array(img):
    lebar, tinggi = img.size
    pixels = list(img.getdata())
    return [pixels[i * lebar:(i + 1) * lebar] for i in range(tinggi)]

def rgb_ke_hsv_pixel(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    maks = max(r, g, b)
    mins = min(r, g, b)
    delta = maks - mins

    if delta == 0:
        h = 0
    elif maks == r:
        h = (60 * ((g - b) / delta) + 360) % 360
    elif maks == g:
        h = (60 * ((b - r) / delta) + 120) % 360
    elif maks == b:
        h = (60 * ((r - g) / delta) + 240) % 360

    return h

def hue_in_range(h, start, end):
    if start <= end:
        return start <= h <= end
    else:
        # rentang hue yang melingkar, misal 350-10 derajat
        return h >= start or h <= end

def segmentasi_hue_multi(img, hue_ranges):
    rgb_array = gambar_ke_array(img)
    mask = []

    for baris in rgb_array:
        baris_mask = []
        for r, g, b in baris:
            h = rgb_ke_hsv_pixel(r, g, b)
            cocok = any(hue_in_range(h, start, end) for (start, end) in hue_ranges)
            baris_mask.append(1 if cocok else 0)
        mask.append(baris_mask)
    return mask


def segmentasi_gelap(img, ambang=60):
    gray_mask = []
    gray_array = gambar_ke_array(img)
    for baris in gray_array:
        baris_mask = []
        for r, g, b in baris:
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            baris_mask.append(1 if gray <= ambang else 0)
        gray_mask.append(baris_mask)
    return gray_mask

def gabungkan_mask(mask1, mask2):
    hasil = []
    for i in range(len(mask1)):
        baris = []
        for j in range(len(mask1[0])):
            baris.append(1 if (mask1[i][j] or mask2[i][j]) else 0)
        hasil.append(baris)
    return hasil

def erosi(mask, kernel_size=3):
    offset = kernel_size // 2
    hasil = []
    for i in range(len(mask)):
        baris = []
        for j in range(len(mask[0])):
            jumlah = 0
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < len(mask) and 0 <= nj < len(mask[0]):
                        jumlah += mask[ni][nj]
            total = kernel_size * kernel_size
            baris.append(1 if jumlah == total else 0)
        hasil.append(baris)
    return hasil

def dilasi(mask, kernel_size=3):
    offset = kernel_size // 2
    hasil = []
    for i in range(len(mask)):
        baris = []
        for j in range(len(mask[0])):
            jumlah = 0
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < len(mask) and 0 <= nj < len(mask[0]):
                        jumlah += mask[ni][nj]
            baris.append(1 if jumlah > 0 else 0)
        hasil.append(baris)
    return hasil

def morfologi_lengkap(mask, kernel_size=3):
    hasil_open = dilasi(erosi(mask, kernel_size), kernel_size)
    hasil_close = erosi(dilasi(hasil_open, kernel_size), kernel_size)
    return hasil_close

def segmentasi_overripe_komplit(img):
    mask_hue_overripe = segmentasi_hue_multi(img, [(10, 30), (260, 310)])
    mask_gelap = segmentasi_gelap(img, ambang=80)
    mask_kuning_tua = segmentasi_hue_multi(img, [(40, 90)])

    mask1 = gabungkan_mask(mask_hue_overripe, mask_gelap)
    mask_total = gabungkan_mask(mask1, mask_kuning_tua)

    mask_total_bersih = morfologi_lengkap(mask_total, kernel_size=3)
    return mask_total_bersih

def terapkan_mask(img, mask):
    rgb_array = gambar_ke_array(img)
    hasil = []

    for i in range(len(mask)):
        baris_hasil = []
        for j in range(len(mask[0])):
            if mask[i][j] == 1:
                baris_hasil.append(rgb_array[i][j])
            else:
                baris_hasil.append((0, 0, 0))
        hasil.append(baris_hasil)
    return hasil

def ekstraksi_fitur_warna(img, mask):
    from scipy.stats import skew, kurtosis

    rgb_array = gambar_ke_array(img)
    hue_values = []
    saturation_values = []
    value_values = []

    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] == 1:
                r, g, b = rgb_array[i][j]
                r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
                maks = max(r_norm, g_norm, b_norm)
                mins = min(r_norm, g_norm, b_norm)
                delta = maks - mins

                if delta == 0:
                    h = 0
                elif maks == r_norm:
                    h = (60 * ((g_norm - b_norm) / delta) + 360) % 360
                elif maks == g_norm:
                    h = (60 * ((b_norm - r_norm) / delta) + 120) % 360
                elif maks == b_norm:
                    h = (60 * ((r_norm - g_norm) / delta) + 240) % 360
                hue_values.append(h)

                s = 0 if maks == 0 else delta / maks
                v = maks
                saturation_values.append(s)
                value_values.append(v)

    if hue_values:
        import numpy as np
        rata_rata_hue = np.mean(hue_values)
        stddev_hue = np.std(hue_values)
        skewness_hue = skew(hue_values)
        kurtosis_hue = kurtosis(hue_values)
        mean_sat = np.mean(saturation_values)
        std_sat = np.std(saturation_values)
        mean_val = np.mean(value_values)
        std_val = np.std(value_values)
    else:
        rata_rata_hue = stddev_hue = skewness_hue = kurtosis_hue = 0
        mean_sat = std_sat = mean_val = std_val = 0

    return {
        "mean_hue": rata_rata_hue,
        "std_hue": stddev_hue,
        "skewness_hue": skewness_hue,
        "kurtosis_hue": kurtosis_hue,
        "mean_saturation": mean_sat,
        "std_saturation": std_sat,
        "mean_value": mean_val,
        "std_value": std_val
    }

def segmentasi_ke_grayscale(img, mask):
    rgb_array = gambar_ke_array(img)
    grayscale_array = []

    for i in range(len(mask)):
        baris = []
        for j in range(len(mask[0])):
            if mask[i][j] == 1:
                r, g, b = rgb_array[i][j]
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            else:
                gray = 0
            baris.append(gray)
        grayscale_array.append(baris)
    return grayscale_array

def hitung_glcm(grayscale_array, level=8):
    glcm = [[0 for _ in range(level)] for _ in range(level)]
    tinggi = len(grayscale_array)
    lebar = len(grayscale_array[0])

    def quantize(gray):
        return min(gray * level // 256, level - 1)

    for i in range(tinggi):
        for j in range(lebar - 1):
            g1 = quantize(grayscale_array[i][j])
            g2 = quantize(grayscale_array[i][j + 1])
            glcm[g1][g2] += 1

    total = sum(sum(row) for row in glcm)
    if total == 0:
        return [[0]*level for _ in range(level)]
    return [[glcm[i][j] / total for j in range(level)] for i in range(level)]

def ekstrak_fitur_glcm(glcm):
    contrast = 0
    energy = 0
    homogeneity = 0
    correlation = 0
    n = len(glcm)

    mean_i = sum(i * sum(glcm[i]) for i in range(n))
    mean_j = sum(j * sum(glcm[i][j] for i in range(n)) for j in range(n))

    std_i = (sum(sum(glcm[i][j] * (i - mean_i)**2 for j in range(n)) for i in range(n)))**0.5
    std_j = (sum(sum(glcm[i][j] * (j - mean_j)**2 for j in range(n)) for i in range(n)))**0.5

    for i in range(n):
        for j in range(n):
            p = glcm[i][j]
            contrast += p * ((i - j) ** 2)
            energy += p ** 2
            homogeneity += p / (1 + abs(i - j))
            if std_i != 0 and std_j != 0:
                correlation += ((i - mean_i) * (j - mean_j) * p) / (std_i * std_j)

    return {
        "contrast": contrast,
        "energy": energy,
        "homogeneity": homogeneity,
        "correlation": correlation
    }

def ekstraksi_fitur_dari_gambar(img, label=None):
    img = img.resize((224, 224)).convert("RGB")
    if label == "ripe":
        mask = segmentasi_hue_multi(img, [(30, 60)])   # kuning
    elif label == "unripe":
        mask = segmentasi_hue_multi(img, [(70, 130)])  # hijau
    elif label == "overripe":
        mask = segmentasi_overripe_komplit(img)
    else:
        mask = [[1]*img.size[0] for _ in range(img.size[1])]

    fitur_warna = ekstraksi_fitur_warna(img, mask)
    gray_array = segmentasi_ke_grayscale(img, mask)
    glcm = hitung_glcm(gray_array)
    fitur_glcm = ekstrak_fitur_glcm(glcm)

    fitur = {
        **fitur_warna,
        **fitur_glcm
    }
    return fitur

def segmentasi_overripe_komplit(img):
    mask_hue_overripe = segmentasi_hue_multi(img, [(0, 20), (330, 360)])  # oranye-merah coklat
    mask_gelap = segmentasi_gelap(img, ambang=80)
    mask_kuning_tua = segmentasi_hue_multi(img, [(40, 90)])  # kuning tua

    mask1 = gabungkan_mask(mask_hue_overripe, mask_gelap)
    mask_total = gabungkan_mask(mask1, mask_kuning_tua)

    mask_total_bersih = morfologi_lengkap(mask_total, kernel_size=3)
    return mask_total_bersih

