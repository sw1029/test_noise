from .base import Denoise
import numpy as np
import cv2
from algotom.prep.removal import remove_stripe_based_wavelet_fft

class DenoiseStripe(Denoise):
    @staticmethod
    def denoise(src: np.ndarray, horizontal: bool = False) -> np.ndarray:
        '''
        입력된 이미지에 노이즈를 보정하는 부분입니다.
        init에서 입력받은 이미지에 noise를 보정한 후 반환합니다. 
        stripe noise에 속하는 noise를 보정합니다.

        푸리에 변환을 사용하며, 실제 코드 구현은 openCV 튜토리얼 코드를 참고하였습니다. 

        참고 레퍼런스
        - https://earthinversion.com/techniques/signal-denoising-using-fast-fourier-transform/
        - https://docs.opencv.org/4.x/d2/d0b/tutorial_periodic_noise_removing_filter.html
        - https://stackoverflow.com/questions/60506244/removal-of-horizontal-stripes-using-opencv2
        '''
        if len(src.shape) == 3 and src.shape[2] == 3:
            b, g, r = cv2.split(src)
            denoised_b = DenoiseStripe.denoise_single(b, horizontal)
            denoised_g = DenoiseStripe.denoise_single(g, horizontal)
            denoised_r = DenoiseStripe.denoise_single(r, horizontal)

            return cv2.merge([denoised_b, denoised_g, denoised_r])
        elif len(src.shape) == 2:
            return DenoiseStripe.denoise_single(src, horizontal)        

    @staticmethod
    def denoise_single(src: np.ndarray, horizontal: bool = False) -> np.ndarray:
        """단일 채널 노이즈 제거를 수행합니다."""
        if len(src.shape) != 2:
            raise ValueError("Input must be a single-channel image.")

        img = src.astype(np.float32)
        # 짝수 크기 요구에 대비하여 '패딩 후 처리, 처리 후 원복'으로 변경
        h0, w0 = img.shape
        pad_bottom = h0 % 2
        pad_right = w0 % 2
        if pad_bottom or pad_right:
            img = cv2.copyMakeBorder(
                img,
                top=0,
                bottom=pad_bottom,
                left=0,
                right=pad_right,
                borderType=cv2.BORDER_REPLICATE,
            )

        # PSD 계산
        psd = DenoiseStripe.calc_psd(img)
        psd = DenoiseStripe.fftshift(psd)
        
        cy, cx = psd.shape[0] // 2, psd.shape[1] // 2
        
        mask = np.ones_like(psd, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), 30, 0, -1)
        mask = psd * mask

        if horizontal:
            line = mask[:, cx]
        else:
            line = mask[cy, :]

        peaks = np.where(line > np.mean(line) + 3 * np.std(line))[0]

        H = np.ones(img.shape, dtype=np.float32)
        radius = 10

        if horizontal:
            for peak in peaks:
                if abs(peak - cy) > 5:
                    DenoiseStripe.synthesize_filter_H(H, (cx, peak), radius)
        else:
            for peak in peaks:
                if abs(peak - cx) > 5:
                    DenoiseStripe.synthesize_filter_H(H, (peak, cy), radius)

        H = DenoiseStripe.fftshift(H)
        img = DenoiseStripe.filter2D_frequency(img, H)

        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

        # 처리 후 원본 크기로 크롭하여 반환
        if (img.shape[0] != h0) or (img.shape[1] != w0):
            img = img[:h0, :w0]
        return img

    
    @staticmethod
    def fftshift(input_image):
        output_image = input_image.copy()
        cx = output_image.shape[1] // 2
        cy = output_image.shape[0] // 2
        q0 = output_image[0:cy, 0:cx]
        q1 = output_image[0:cy, cx:cx * 2]
        q2 = output_image[cy:cy * 2, 0:cx]
        q3 = output_image[cy:cy * 2, cx:cx * 2]

        tmp = q0.copy()
        q0[:] = q3
        q3[:] = tmp
        tmp = q1.copy()
        q1[:] = q2
        q2[:] = tmp

        return output_image

    @staticmethod
    def filter2D_frequency(input_image, filter_image):
        planes = [input_image.astype(np.float32), np.zeros(input_image.shape, dtype=np.float32)]
        complex_image = cv2.merge(planes)
        cv2.dft(complex_image, complex_image, flags=cv2.DFT_SCALE)
        planes_filter = [filter_image.astype(np.float32), np.zeros(filter_image.shape, dtype=np.float32)]
        complex_filter = cv2.merge(planes_filter)
        complex_image_shifted = DenoiseStripe.fftshift(complex_image)
        complex_image_filter = cv2.mulSpectrums(complex_image_shifted, complex_filter, 0)
        complex_image_filter_shifted_back = DenoiseStripe.fftshift(complex_image_filter)

        idft_image = cv2.idft(complex_image_filter_shifted_back)
        planes_output = cv2.split(idft_image)
        output_image = planes_output[0]

        return output_image

    @staticmethod
    def synthesize_filter_H(filter_H, center, radius):
        rows, cols = filter_H.shape
        crow, ccol = rows // 2, cols // 2
        cv2.circle(filter_H, center, radius, 0, -1)
        sym_center = (cols - center[0], rows - center[1])
        cv2.circle(filter_H, sym_center, radius, 0, -1)


    @staticmethod
    def calc_psd(input_image, flag=0):
        planes = [input_image.astype(np.float32), np.zeros(input_image.shape, dtype=np.float32)]
        complex_image = cv2.merge(planes)
        cv2.dft(complex_image, complex_image)
        planes = cv2.split(complex_image)
        planes[0][0, 0] = 0 
        planes[1][0, 0] = 0

        mag_i = cv2.magnitude(planes[0], planes[1])
        
        if flag:
            mat_log = cv2.log(mag_i + 1)
            return mat_log

        return mag_i**2 

    @staticmethod
    def denoise_algotom(
        src: np.ndarray,
        direction: str = "vertical",
        size: int = 5,
        level: int = 7,
    ) -> np.ndarray:
        '''
        오픈소스인 algotom을 이용하여 denoise를 수행합니다.

        Parameters
        - direction: 'vertical' | 'horizontal' (기본: 'vertical')
          algotom의 제거기는 세로 줄무늬(열 방향) 제거에 최적화되어 있으므로,
          가로 줄무늬의 경우 90° 회전하여 처리 후 되돌립니다.
        - size, level: algotom remove_stripe_based_wavelet_fft 파라미터
        '''
        dir_norm = (direction or "vertical").lower()
        is_horizontal = dir_norm.startswith("h")

        def _pad_to_multiple(img2: np.ndarray, m: int) -> tuple[np.ndarray, int, int]:
            h, w = img2.shape
            h2 = ((h + m - 1) // m) * m
            w2 = ((w + m - 1) // m) * m
            pad_b = h2 - h
            pad_r = w2 - w
            if pad_b or pad_r:
                img2 = cv2.copyMakeBorder(
                    img2, 0, pad_b, 0, pad_r, borderType=cv2.BORDER_REPLICATE
                )
            return img2, pad_b, pad_r

        def _wavelet_denoise_single(ch: np.ndarray) -> np.ndarray:
            m = 1 << max(int(level), 0)
            h0, w0 = ch.shape
            chf = ch.astype(np.float32, copy=False)
            chf, pb, pr = _pad_to_multiple(chf, m)
            out = remove_stripe_based_wavelet_fft(chf, size=size, level=level)
            out = out[:h0, :w0]
            return out

        if len(src.shape) == 3 and src.shape[2] == 3:
            b, g, r = cv2.split(src)

            if is_horizontal:
                b_rot = np.rot90(b, 1)
                g_rot = np.rot90(g, 1)
                r_rot = np.rot90(r, 1)

                db = _wavelet_denoise_single(b_rot)
                dg = _wavelet_denoise_single(g_rot)
                dr = _wavelet_denoise_single(r_rot)

                denoised_b = np.rot90(db, -1)
                denoised_g = np.rot90(dg, -1)
                denoised_r = np.rot90(dr, -1)
            else:
                denoised_b = _wavelet_denoise_single(b)
                denoised_g = _wavelet_denoise_single(g)
                denoised_r = _wavelet_denoise_single(r)

            result = cv2.merge([denoised_b, denoised_g, denoised_r])
            if src.dtype == np.uint8:
                return np.clip(result, 0, 255).astype(np.uint8)
            return result

        elif len(src.shape) == 2:
            if is_horizontal:
                src_rot = np.rot90(src, 1)
                denoised_rot = _wavelet_denoise_single(src_rot)
                denoised_img = np.rot90(denoised_rot, -1)
            else:
                denoised_img = _wavelet_denoise_single(src)

            if src.dtype == np.uint8:
                return np.clip(denoised_img, 0, 255).astype(np.uint8)
            return denoised_img
        else:
            return src 
