import numpy as np
import scipy

def remove_silent_frame(xl, xr, yl, yr, dyn_range, n, k):
    # setup masking parameters
    frames  = np.arange(0, len(xl)-n, k)  # define length of frames
    w       = scipy.signal.windows.hann(n)  # define window (Hanning)
    msk_l   = np.zeros((1, len(frames)))         # masker for left channel
    msk_r   = np.zeros((1, len(frames)))         # masker for right channel

    for j in range(len(frames)-1):
        jj          = np.arange(0, len(frames))
        msk_l[j]    = 20 * np.log10(np.linalg.norm(xl[jj],2) * w) / np.sqrt(n)
        msk_r[j]    = 20 * np.log10(np.linalg.norm(xr[jj],2) * w) / np.sqrt(n)

    msk_l   = msk_l[msk_l-max([msk_l, msk_r])+dyn_range > 0]
    msk_r   = msk_r[msk_r-max([msk_l, msk_r])+dyn_range > 0]
    msk_l   = msk_l or msk_r
    count   = 1

    xl_sil  = np.zeros((1, len(xl)))
    xr_sil  = np.zeros((1, len(xr)))
    yl_sil  = np.zeros((1, len(yl)))
    yr_sil  = np.zeros((1, len(yr)))

    for j in range(len(frames)-1):
        if msk[j]:
            jj_i            = np.arange(frames[j], frames[j]+N-1)
            jj_o            = np.arange(frames[count], frames[count]+N-1)
            xl_sil[jj_o]    = xl_sil[jj_o] + xl[jj_i] * w
            xr_sil[jj_o]    = xr_sil[jj_o] + xr[jj_i] * w
            yl_sil[jj_o]    = yl_sil[jj_o] + yl[jj_i] * w
            yr_sil[jj_o]    = yr_sil[jj_o] + yr[jj_i] * w
            count           = count+1

    xl_sil = xl_sil[0:jj_o[-1]]
    xr_sil = xr_sil[0:jj_o[-1]]
    yl_sil = yl_sil[0:jj_o[-1]]
    yr_sil = yr_sil[0:jj_o[-1]]