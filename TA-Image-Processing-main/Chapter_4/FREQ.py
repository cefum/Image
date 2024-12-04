import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# تابعی برای اعمال فوریه و بازسازی تصویر با تعداد محدودی از فرکانس‌ها
def apply_fourier(image, num_frequencies):
    # تبدیل تصویر به فضای فوریه
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    # استخراج فاز و طیف
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    
    # ترتیب فرکانس‌ها بر اساس شدت
    indices = np.argsort(magnitude.ravel())[::-1]  # مرتب‌سازی از بیشترین به کمترین شدت
    mask = np.zeros_like(magnitude)
    
    # نگه‌داشتن بیشترین فرکانس‌ها
    selected_indices = indices[:num_frequencies]
    
    # اعمال ماسک بر روی فرکانس‌ها
    for i in selected_indices:
        y, x = np.unravel_index(i, magnitude.shape)
        mask[y, x] = 1
    
    # اعمال ماسک به فضای فوریه
    fshift_masked = fshift * mask
    f_masked = np.fft.ifftshift(fshift_masked)
    
    # تبدیل فوریه معکوس برای بازسازی تصویر
    img_reconstructed = np.fft.ifft2(f_masked)
    
    # بازگرداندن تصویر بازسازی‌شده، طیف و فاز
    return np.abs(img_reconstructed), magnitude, phase, selected_indices

# بارگذاری تصویر
image = cv2.imread('c:/ta/lena.bmp', 0)  # تصویر را به صورت خاکی-سفید بارگذاری می‌کنیم

# تنظیمات ابتدایی
initial_frequencies = 100  # تعداد اولیه فرکانس‌ها
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# نمایش تصویر اصلی
ax[0, 0].imshow(image, cmap='gray')
total_frequencies = image.size  # تعداد کل پیکسل‌ها
ax[0, 0].set_title(f'Original Image ({total_frequencies} total frequencies)')
ax[0, 0].axis('off')

# نمایش طیف و فاز اولیه
reconstructed_image, magnitude, phase, selected_indices = apply_fourier(image, initial_frequencies)
ax[0, 1].imshow(np.log(1 + magnitude), cmap='gray')
ax[0, 1].set_title('Magnitude Spectrum')
ax[0, 1].axis('off')

ax[0, 2].imshow(phase, cmap='gray')
ax[0, 2].set_title('Phase Spectrum')
ax[0, 2].axis('off')

# نمایش تصویر بازسازی‌شده با تعداد فرکانس مشخص
ax[1, 0].imshow(reconstructed_image, cmap='gray')
percentage = (initial_frequencies / total_frequencies) * 100
ax[1, 0].set_title(f'Reconstructed Image ({initial_frequencies} frequencies, {percentage:.2f}% used)')
ax[1, 0].axis('off')

# اضافه کردن نقاط منتخب به طیف و فاز
magnitude_copy = np.copy(magnitude)
phase_copy = np.copy(phase)

# تغییر رنگ نقاط منتخب در طیف و فاز
for i in selected_indices:
    y, x = np.unravel_index(i, magnitude.shape)
    magnitude_copy[y, x] = np.max(magnitude)  # تغییر رنگ به بیشترین شدت
    phase_copy[y, x] = np.pi  # تغییر رنگ فاز

ax[1, 1].imshow(np.log(1 + magnitude_copy), cmap='hot')
ax[1, 1].set_title('Magnitude Spectrum with Selected Frequencies')
ax[1, 1].axis('off')

ax[1, 2].imshow(phase_copy, cmap='hot')
ax[1, 2].set_title('Phase Spectrum with Selected Frequencies')
ax[1, 2].axis('off')

# ایجاد اسلایدر برای تغییر تعداد فرکانس‌ها
ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Num Frequencies', 1, 10000, valinit=initial_frequencies, valstep=1)

# تابعی برای به‌روزرسانی تصویر هنگام تغییر اسلایدر
def update(val):
    num_frequencies = int(slider.val)
    
    # اعمال تبدیل فوریه و بازسازی تصویر
    reconstructed_image, magnitude, phase, selected_indices = apply_fourier(image, num_frequencies)
    
    # بروزرسانی تصاویر
    ax[1, 0].imshow(reconstructed_image, cmap='gray')
    percentage = (num_frequencies / total_frequencies) * 100
    ax[1, 0].set_title(f'Reconstructed Image ({num_frequencies} frequencies, {percentage:.2f}% used)')    
    ax[0, 1].imshow(np.log(1 + magnitude), cmap='gray')
    ax[0, 1].set_title('Magnitude Spectrum')
    
    ax[0, 2].imshow(phase, cmap='gray')
    ax[0, 2].set_title('Phase Spectrum')
    
    # اضافه کردن نقاط منتخب به طیف و فاز
    magnitude_copy = np.copy(magnitude)
    phase_copy = np.copy(phase)
    
    for i in selected_indices:
        y, x = np.unravel_index(i, magnitude.shape)
        magnitude_copy[y, x] = np.max(magnitude)  # تغییر رنگ به بیشترین شدت
        phase_copy[y, x] = np.pi  # تغییر رنگ فاز

    ax[1, 1].imshow(np.log(1 + magnitude_copy), cmap='hot')
    ax[1, 1].set_title('Magnitude Spectrum with Selected Frequencies')
    
    ax[1, 2].imshow(phase_copy, cmap='hot')
    ax[1, 2].set_title('Phase Spectrum with Selected Frequencies')
    
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
