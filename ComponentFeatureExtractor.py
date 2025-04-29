import numpy as np
import cv2

# Define the kernel offsets corresponding to neighbors q0 to q7
# Orientation: 
# q0 q1 q2
# q3  p q4
# q5 q6 q7
kernel_offsets = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]

# 8 neighbors, N = 8
N = 8

class ComponentFeatureExtractor:
    def __init__(self, T0=5, T1=10, G0=0.1):
        self.T0 = T0
        self.T1 = T1
        self.G0 = G0

    def _compute_ppv(self, component):
        # (Padding the component with 1 pixel border)
        padded = np.pad(component, ((1, 1), (1, 1)), mode='constant', constant_values=0)
        h, w = component.shape
        ppv = np.zeros((h, w, N), dtype=np.uint8)

        # (Accessing neighboring pixel values using offsets)
        for i in range(h):
            for j in range(w):
                center = padded[i+1, j+1]
                for idx, (dy, dx) in enumerate(kernel_offsets):
                    neighbor = padded[i+1+dy, j+1+dx]
                    diff = int(center) - int(neighbor)

                    # (Eq.1) Penta Pattern Vector (PPV)
                    if abs(diff) < self.T0:
                        ppv[i, j, idx] = 0
                    elif self.T0 <= diff < self.T1:
                        ppv[i, j, idx] = 1
                    elif -self.T1 < diff <= -self.T0:
                        ppv[i, j, idx] = 2
                    elif diff >= self.T1:
                        ppv[i, j, idx] = 3
                    elif diff <= -self.T1:
                        ppv[i, j, idx] = 4
        return ppv

    def _compute_bpvs(self, ppv):
        h, w, _ = ppv.shape
        bpvs = np.zeros((h, w, 5, N), dtype=np.uint8)

        for k in range(5):
            # (Eq.2) BPVc,N,k(p)
            bpvs[:, :, k, :] = (ppv == k).astype(np.uint8)

        return bpvs

    def _compute_uniformity(self, bpv):
        # (Eq.3) Uniformity for BPVs
        return np.sum(np.abs(np.diff(bpv, axis=-1)), axis=-1)

    def _compute_ei(self, component):
        padded = np.pad(component, ((1,1), (1,1)), mode='constant', constant_values=0)
        h, w = component.shape
        EI = np.zeros((h, w, 5), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                p = padded[i+1, j+1]
                q = [padded[i+1+dy, j+1+dx] for dy, dx in kernel_offsets]

                # (Eq.4) EI0
                EI0 = int(abs(int(p) - int(q[3])) <= self.T0 and abs(int(p) - int(q[4])) <= self.T0)

                # (Eq.5) EI1
                EI1 = int(abs(int(p) - int(q[1])) <= self.T0 and abs(int(p) - int(q[6])) <= self.T0)

                # (Eq.6) EI2
                EI2 = int(abs(int(p) - int(q[2])) <= self.T0 and abs(int(p) - int(q[5])) <= self.T0)

                # (Eq.7) EI3
                EI3 = int(abs(int(p) - int(q[0])) <= self.T0 and abs(int(p) - int(q[7])) <= self.T0)

                # (Eq.8) EI4
                EI4 = int((EI0 + EI1 + EI2 + EI3) == 0)

                EI[i, j] = [EI0, EI1, EI2, EI3, EI4]

        return EI

    def _compute_gabor_features(self, component):
        h, w = component.shape
        # (Applying Gabor filters)
        gabor_kernels = []
        for theta in [0, np.pi/2]:
            for scale in range(3):
                ksize = 10
                sigma = 4.0
                lambd = 10.0
                gamma = 0.5
                psi = 0
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
                gabor_kernels.append(kernel)

        gabor_responses = []
        for kernel in gabor_kernels:
            filtered = cv2.filter2D(component, cv2.CV_32F, kernel)
            gabor_responses.append(filtered)

        # Split responses
        psi_m0 = gabor_responses[0:3]  # 0 degree
        psi_m90 = gabor_responses[3:6]  # 90 degree

        return psi_m0, psi_m90

    def _compute_gradient_orientation_similarity(self, psi_m0, psi_m90):
        h, w = psi_m0[0].shape
        Gm = []

        for m in range(3):
            padded_0 = np.pad(psi_m0[m], ((1,1), (1,1)), mode='constant', constant_values=0)
            padded_90 = np.pad(psi_m90[m], ((1,1), (1,1)), mode='constant', constant_values=0)
            G = np.zeros((h,w), dtype=np.float32)

            for i in range(h):
                for j in range(w):
                    # (Eq.9) Gm(p)
                    numerator = abs(padded_0[i+1, j+2] - padded_0[i+1, j+1])
                    denominator = abs(padded_90[i, j+1] - padded_90[i+1, j+1])
                    if denominator == 0:
                        denominator = 1e-6  # To avoid division by zero
                    G[i, j] = np.arctan(numerator / denominator)

            Gm.append(G)

        return Gm

    def _compute_egm(self, Gm):
        h, w = Gm[0].shape
        EG = []

        for m in range(3):
            padded = np.pad(Gm[m], ((1,1),(1,1)), mode='constant', constant_values=0)
            EGm = np.zeros((h, w, 5), dtype=np.uint8)

            for i in range(h):
                for j in range(w):
                    p = padded[i+1, j+1]
                    q = [padded[i+1+dy, j+1+dx] for dy, dx in kernel_offsets]

                    # (Eq.10) EG0_m
                    EG0 = int(abs(p - q[3]) <= self.G0 and abs(p - q[4]) <= self.G0)

                    # (Eq.11) EG1_m
                    EG1 = int(abs(p - q[1]) <= self.G0 and abs(p - q[6]) <= self.G0)

                    # (Eq.12) EG2_m
                    EG2 = int(abs(p - q[2]) <= self.G0 and abs(p - q[5]) <= self.G0)

                    # (Eq.13) EG3_m
                    EG3 = int(abs(p - q[0]) <= self.G0 and abs(p - q[7]) <= self.G0)

                    # (Eq.14) EG4_m
                    EG4 = int((EG0 + EG1 + EG2 + EG3) == 0)

                    EGm[i,j] = [EG0, EG1, EG2, EG3, EG4]

            EG.append(EGm)

        return EG

    def _compute_final_line_orientation(self, EI, EG):
        h, w, _ = EI.shape
        E_final = []

        for m in range(3):
            E_l = np.zeros((h, w, 5), dtype=np.uint8)

            for l in range(4):
                # (Eq.15) E^l_m(p)
                E_l[:,:,l] = (EI[:,:,l] == EG[m][:,:,l]).astype(np.uint8) * EI[:,:,l]

            # (Eq.16) E4_m(p)
            E_l[:,:,4] = ((E_l[:,:,0] + E_l[:,:,1] + E_l[:,:,2] + E_l[:,:,3]) == 0).astype(np.uint8)

            E_final.append(E_l)

        return E_final

    def _bpv_to_bin(self, bpv):
        # Helper: Convert binary vector to bin id (uniform LBP mapping)
        bpv = bpv.astype(np.uint8)
        decimal = bpv.dot(1 << np.arange(bpv.shape[-1]-1, -1, -1))
        return decimal % 59  # Mapping to 59 uniform patterns

    def extract_features(self, component):
        ppv = self._compute_ppv(component)
        bpvs = self._compute_bpvs(ppv)
        EI = self._compute_ei(component)
        psi_m0, psi_m90 = self._compute_gabor_features(component)
        Gm = self._compute_gradient_orientation_similarity(psi_m0, psi_m90)
        EG = self._compute_egm(Gm)
        E_final = self._compute_final_line_orientation(EI, EG)

        h, w, _ = ppv.shape
        F1 = np.zeros((3, 5, 5, 59), dtype=np.float32)

        for m in range(3):
            for k in range(5):
                for l in range(5):
                    for i in range(h):
                        for j in range(w):
                            if E_final[m][i,j,l] == 1:
                                bin_idx = self._bpv_to_bin(bpvs[i,j,k,:])
                                F1[m,k,l,bin_idx] += 1

        # (Eq.22) F1m,k,l(d)

        # BMPV calculation
        BMPV = []

        for m in range(3):
            padded_phi_h = np.pad(psi_m0[m], ((1,1),(1,1)), mode='constant')
            padded_phi_v = np.pad(psi_m90[m], ((1,1),(1,1)), mode='constant')

            phi = np.sqrt(np.square(padded_phi_h[1:-1,1:-1]) + np.square(padded_phi_v[1:-1,1:-1]))

            bmpv = np.zeros((h,w,N), dtype=np.uint8)

            for i in range(h):
                for j in range(w):
                    p = phi[i,j]
                    neighbors = [phi[i+dy, j+dx] for dy, dx in kernel_offsets]

                    for idx, neighbor in enumerate(neighbors):
                        # (Eq.17) BMPVm(p)
                        bmpv[i,j,idx] = int(neighbor >= p)

            BMPV.append(bmpv)

        # (Eq.18) FBMPV: Normalize histogram over BMPV
        FBMPV = []

        for m in range(3):
            hist = np.sum(BMPV[m], axis=(0,1))
            hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
            FBMPV.append(hist)

        return F1, FBMPV
