# This is class file for learning fast PAT


import numpy as np
from scipy.interpolate import griddata
import h5py


def extract_kgrid(filename,imageName):
  """Extract the images into a 3D uint8 numpy array [index, y, x, depth]."""
  fData = h5py.File(filename,'r')
  inData = fData.get(imageName)  
      
  rows = inData.shape[0]
  cols = inData.shape[1]
#  deps = inData.shape[2]
  print(rows, cols)
  data = np.array(inData)
    
  data = data.reshape(rows, cols)
  #shape into Matlab form!
  data = np.transpose(data,(1,0))
  return data



class kgrid(object):
    def __init__(self,KGRID_NAME):
        
        kx=extract_kgrid(KGRID_NAME,'kx')
        ky=extract_kgrid(KGRID_NAME,'ky')
        k=extract_kgrid(KGRID_NAME,'k')
        #Hard code some dimensions, for now
        self.dx = 0.1e-3
        self.dy = 0.1e-3
        self.c  = 1500.0
        self.dt = (2/3)*1e-7
        self.geometry_scaling=1.0 #Not needed for now
        self.time_scaling=1.0  #Not needed for now
        
        
        self._kx=kx
        self._ky=ky
        self._k=k
        
        
    @property
    def kx(self):        
         return self._kx
        
    @property
    def ky(self):        
         return self._ky
     
    @property
    def k(self):        
         return self._k
             
        

class fastPAT(object):
    def __init__(self, kgridBack,kgridForw,angTresh):
        
        
        #Load kgrids
        self._kgridBack=kgridBack
        self._kgridForw=kgridForw
        self._angThresh=angTresh

        
    @property
    def kgridBack(self):        
         return self._kgridBack
     

    @property
    def kgridForw(self):        
         return self._kgridForw

#    @property
#    def kgridFew(self):        
#         return self._kgridFew
        
#    @property
#    def kgrid_sampled(self):
#         return self._kgrid_sampled
     
    @property
    def angThresh(self):
        return self._angThresh
    

    def kspace_forward(self,p0):
        
        
        c=self.kgridForw.c
#        p0 = zeros(Nx,Ny);
#        
#        p0(1:Nx,1:Nx) = p0_in;
#        
#        % make p0 symmetrical about y = 0
#        p0 = (p0 + fliplr(p0))/2;
        p0=np.transpose(p0[::-1,:],(1,0))
        p0=np.concatenate((p0[:,:],p0[:,::-1]),1)

#        p_kxky = fftshift(fftn(ifftshift(p0)));
        p_kxky=np.fft.fftshift(np.fft.fftn(np.fft.fftshift(p0)))
#        % create a computational grid that is evenly spaced in kx and w_new
#        w_new = c .* kgrid.ky;
        w_new = c*self.kgridForw.ky
#        
#        % calculate the values of ky corresponding to this (w,kx) grid
#        ky_new = real(sign(w_new).*sqrt( (w_new/c).^2 - kgrid.kx.^2));
        ky_new= np.real( np.sign(w_new)*np.sqrt( np.square(w_new/c) - np.square( self.kgridForw.kx) )    )
#        % interpolation from regular (kx, ky) grid to regular (kx, w_new) grid
#        p_kxw = interpn(kgrid.kx, kgrid.ky, p_kxky, kgrid.kx, ky_new, 'linear');
#        

        kxDim=np.reshape(self.kgridForw.kx,[128*64,1])
        kyDim=np.reshape(self.kgridForw.ky,[128*64,1])
        kyNewDim=np.reshape(ky_new,[128*64,1])
                
        ptsEval=np.concatenate((kxDim,kyDim),1)
        ptsInterp=np.concatenate((kxDim,kyNewDim),1)
        
        p_kxw   = griddata(ptsEval,np.reshape(p_kxky,[128*64,1]),ptsInterp,method='linear') #Should be nearest but throws error! :/
        p_kxw=np.reshape(p_kxw,[64,128])
#        % set values outside the interpolation range to zero
#        p_kxw(isnan(p_kxw)) = 0;
        idx  = np.where(np.isnan(p_kxw))
        p_kxw[idx]=0
        
#        % remove any evanescent parts of the field (keeping the dc term)
#        p_kxw = p_kxw.*(ky_new~=0);
#        p_kxw((w_new==0)&(kgrid.kx==0)) = p_kxky((kgrid.ky==0)&(kgrid.kx==0)); #Note w_new = c*ky
        idx  = np.where(ky_new == 0)
        p_kxw[idx]=0
        
        idx=np.where((np.abs(w_new)+np.abs(self.kgridForw.kx)) == 0)        
        p_kxw[idx] = p_kxky[idx]
        
#        % calculate the regularised weighting factor %
#        max_angle = angIn;                             % [degrees]
#        min_ky = abs((w_new/c)*cosd(max_angle));    % ky values corresponding to max_angle
#        wf = w_new ./ (c * ky_new);                 % unregularised weighting factor
#        wf(isnan(wf)) = 0;                          % removing NaNs (when w_new = ky_new = 0)
#        wf(abs(ky_new) < min_ky) = 0;               % removing components beyond max_angle
#        wf(w_new==0 & kgrid.kx==0) = 1;             % keep the DC component
#        p_kxw_sf = wf .* p_kxw;                     % apply the weighting factor
        max_angle = self.angThresh
        min_ky = np.abs( (w_new/c) * np.cos( np.deg2rad(max_angle) ) )
        wf = w_new / (c * ky_new)
        idx  = np.where(np.isnan(wf))
        wf[idx]=0
        idx  = np.where(np.abs(ky_new) < min_ky)
        wf[idx]=0
        idx  = np.where(np.abs(w_new) + np.abs(self.kgridForw.kx) == 0)
        wf[idx]=1
        p_kxw_sf = wf * p_kxw;
#        
#        % compute the inverse FFT of p(kx, ky) to yield p(x, y)
#        p_ty = real(fftshift(ifftn(ifftshift(p_kxw_sf))));
        pT=np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(p_kxw_sf))))
#        % take result for positive times (as it has been designed to be symmetric)
#        p_approx = p_ty(:,end/2+1:end);
#        pfast_kspace=p_approx(end:-1:1,:)';
        pT=pT[:,64:128]
        pT=np.transpose(pT[:,:],(1,0))
 
        return (pT)

    def kspace_backward(self,pT):
        

        c=self.kgridBack.c

        #   p = [flipdim(p, 1); p(2:end, :)];        
        #Transpose frm Python dimensions, maybe not needed later

        #Might be needed again, currently using k_wave output
#        pT=np.transpose(pT,(1,0))        
        pT=np.concatenate((pT[::-1,:],pT[1::,:]),0)
        
        #        w = c .* kgrid.kx;
        #        w_new = c .* kgrid.k;
        w=c*self.kgridBack.kx
        w_new = c*self.kgridBack.k
        
        #        sf = c.^2 .* sqrt( (w ./ c).^2 - kgrid.ky.^2) ./ (2 .* w);
        sf=c*c*np.sqrt(  np.square(w/c) - np.square(self.kgridBack.ky) )
        sf=np.divide(sf , 2*w)
        
        #      sf(w == 0 & kgrid.ky == 0) = c ./ 2;
        idx=np.where((np.abs(w)+np.abs(self.kgridBack.ky)) == 0)
        sf[idx]=c/2.0

        #        p = sf .* fftshift(fftn(ifftshift(p)));
        pT=np.fft.fftshift(np.fft.fftn(np.fft.fftshift(pT)))
        pT=np.multiply(sf,pT)
        
        #       p(abs(w) < abs(c * kgrid.ky)) = 0;
        idx=np.where(np.abs(w) <  (c*self.kgridBack.ky) )
        pT[idx]=0.0

        
        wDim=np.reshape(w,[127*64,1])
        kyDim=np.reshape(self.kgridBack.ky,[127*64,1])
        wNewDim=np.reshape(w_new,[127*64,1])
        
        ptsEval=np.concatenate((kyDim,wDim),1)
        ptsInterp=np.concatenate((kyDim,wNewDim),1)
#        p = interp2(kgrid.ky, w, p, kgrid.ky, w_new, interp_method);        
#        p(isnan(p)) = 0;

        p0   = griddata(ptsEval,np.reshape(pT,[127*64,1]),ptsInterp,method='linear') #To be consistent with above
        idx  = np.where(np.isnan(p0))
        p0[idx]=0.0
        
        p0=np.reshape(p0,[127,64])
        
#        p = real(fftshift(ifftn(ifftshift(p))));
        p0=np.real(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(p0))))
        

#        p = 2 * 2 * p ./ c;         
        p0=(2*2/c)*p0
    
        p0 = p0[64-1:128,:];        
        
        
        return (p0)
    
    def subSample(self):
        return(self.p0)



def init_kspace():
  class kSpaceMethods(object):
    pass
  kspace = kSpaceMethods()
  
  c=int(1500)  

#  data_sets.train = DataSet(train_images, train_true, train_grad)
  kspace.test = fastPAT(c)

  return kspace