import os
import gc
import time
import h5py
import numpy as np
import json
from pathlib import Path
from suite2p import run_s2p
from ScanImageTiffReader import ScanImageTiffReader
import multiprocessing
import matplotlib.pyplot as plt
import ast

class ImageProcessor:
    def __init__(self, tiff_folder_list=[], exp_list=[], x_bounds=[0,512], y_bounds=[0,512],
                 out_path='out_path', channels=1, channelOI=[0],cropToHoloFOV=0):

        if len(tiff_folder_list)==0:
            print('Need at least one tif folder to analyze')
            return
        self.tiff_folder_list = tiff_folder_list
        if not exp_list:
            exp_list = [t.split('\\')[-1].replace('//','').replace( r'/' ,'').replace(r'\\','') for t in tiff_folder_list]
            print(f'assigning .h5 names: {exp_list}')
        self.exp_list = exp_list
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.out_path = out_path
        self.channels = channels
        self.channelOI = channelOI
        self.outdirs=[]
        self.cropToHoloFOV = cropToHoloFOV

    # Helper
    def create_directory(self,path):
        if not os.path.exists(path):
            try:
                print(f'\n Creating directory: {path}')
                os.makedirs(path)
            except:
                print('Cannot create directory.. maybe the name is wrong?')
        else:
            print('Directory already exists')

   # -1 meso
    def get_metadata_info(self):
        """ this module was adapted (mostly copied) from a suite2p github issue --> https://github.com/MouseLand/suite2p/issues/802
              converting to python an existent matlab code for mesoscopic data.
            the goal is to get the XY boundaries of each MROI in ScanImage for mesoscopic data processing"""

        print(f'searching the metadata info of the first tif')
        image_path=str(list(Path(self.tiff_folder_list[0]).glob('*.tif'))[0])
        image_file =  ScanImageTiffReader(image_path)
        metadata_raw = image_file.metadata()
        image = image_file.data()
        metadata_str = metadata_raw.split('\n\n')[0]
        metadata_json = metadata_raw.split('\n\n')[1]
        metadata_dict = dict(item.split('=') for item in metadata_str.split('\n') if 'SI.' in item)
        metadata = {k.strip().replace('SI.','') : v.strip() for k, v in metadata_dict.items()}
        for k in list(metadata.keys()):
            if '.' in k:
                ks = k.split('.')
                # TODO just recursively create dict from .-containing values
                if k.count('.') == 1:
                    if not ks[0] in metadata.keys():
                        metadata[ks[0]] = {}
                    metadata[ks[0]][ks[1]] = metadata[k]
                elif k.count('.') == 2:
                    if not ks[0] in metadata.keys():
                        metadata[ks[0]] = {}
                        metadata[ks[0]][ks[1]] = {}
                    elif not ks[1] in metadata[ks[0]].keys():
                        metadata[ks[0]][ks[1]] = {}
                    metadata[ks[0]][ks[1]] = metadata[k]
                elif k.count('.') > 2:
                    print('skipped metadata key ' + k + ' to minimize recursion in dict')
                metadata.pop(k)
        metadata['json'] = json.loads(metadata_json)
        frame_rate = metadata['hRoiManager']['scanVolumeRate']
        if 'userZs' in metadata['hFastZ']:
            z_collection = metadata['hFastZ']['userZs']
            num_planes = len(z_collection)
        else:
            num_planes=1
        roi_metadata = metadata['json']['RoiGroups']['imagingRoiGroup']['rois']
        zs = ast.literal_eval(metadata['hStackManager']['zs'].replace(" ",","))
        # Extract ROI dimensions and information for each z-plane
        roi = {}
        w_px = []
        h_px = []
        cXY = []
        szXY = []
        if (type(roi_metadata) == list):
            num_rois = len(roi_metadata)
            for r in range(num_rois):
                    roi[r] = {}
                    roi[r]['w_px'] = roi_metadata[r]['scanfields']['pixelResolutionXY'][0]
                    w_px.append(roi[r]['w_px'])
                    roi[r]['h_px'] = roi_metadata[r]['scanfields']['pixelResolutionXY'][1]
                    h_px.append(roi[r]['h_px'])
                    roi[r]['center'] = roi_metadata[r]['scanfields']['centerXY']
                    cXY.append(roi[r]['center'])
                    roi[r]['size'] = roi_metadata[r]['scanfields']['sizeXY']
                    szXY.append(roi[r]['size'])
                    #print('{} {} {}'.format(roi[r]['w_px'], roi[r]['h_px'], roi[r]['size']))
            w_px = np.asarray(w_px)
            h_px = np.asarray(h_px)
            szXY = np.asarray(szXY)
            cXY = np.asarray(cXY)
            cXY = cXY - szXY / 2
            cXY = cXY - np.amin(cXY, axis=0)
            mu = np.median(np.transpose(np.asarray([w_px, h_px])) / szXY, axis=0)
            imin = cXY * mu

            #deduce flyback time
            n_rows_sum = np.sum(h_px)
            n_flyback = (image_file.shape()[1] - n_rows_sum) / np.max([1, num_rois - 1])

            irow = np.insert(np.cumsum(np.transpose(h_px) + n_flyback), 0, 0)
            irow = np.delete(irow, -1)
            irow = np.vstack((irow, irow + np.transpose(h_px)))

        elif (type(roi_metadata)==dict):
            num_rois = len(np.unique(zs ))

        data = {}
        data['fs'] = frame_rate
        data['nplanes'] =len(np.unique(zs ))# To do: check that this is a general property for finding if a recording configuration is multiplane or not
        data['nrois'] = num_rois #or irow.shape[1]?
        if data['nrois'] == 1:
            data['mesoscan'] = 0
            data['multiplane'] = 0
        elif data['nplanes']==1 and data['nrois']>1:
            data['mesoscan'] = 1
            data['multiplane'] = 0
        elif data['nplanes']>1 and data['nrois']>1:
            data['mesoscan'] = 0
            data['multiplane'] = 1

        if data['mesoscan']:
            data['dx'] = []
            data['dy'] = []
            data['lines'] = []
            for i in range(num_rois):
                data['dx'] = np.hstack((data['dx'], imin[i,0]))
                data['dy'] = np.hstack((data['dy'], imin[i,1]))
                data['lines'].append(list(range(irow[0,i].astype('int32'), irow[1,i].astype('int32')))) ### TODO NOT QUITE RIGHT YET
            data['lines'] = np.stack(data['lines'])
            data['dx'] = data['dx'].astype('int32')
            data['dy'] = data['dy'].astype('int32')
            data['w_px'] = w_px
            data['h_px'] = h_px
        self.meso_params = data
        self.image = image
        self.planes = data['nplanes']
        print(f'found {self.planes} planes and {num_rois} independent ROIs')
        return metadata

    # Optional 0
    def plotReconstruction(self):
        rois = []
        if self.meso_params['mesoscan']:
            for i in range(self.meso_params['nrois']):
                i_stripe = slice(i,self.image.shape[0],self.meso_params['nrois'])
                rois.append(self.image[i_stripe,self.meso_params['lines'][i],:])
        elif self.meso_params['multiplane']:
            for i in range(self.planes):
                i_stripe = slice(i,self.image.shape[0],self.planes)
                rois.append(self.image[i_stripe,:,:])
        else:
            rois = [self.image]

        meanrois = [np.mean(i,0) for i in rois]
        print('Concatenating ROIs to verify image reconstruction: ')
        reconstruction = np.hstack(meanrois)

        mimg1 = np.percentile(reconstruction,0.01)
        mimg99 = np.percentile(reconstruction,99.9)
        mscaled = (reconstruction - mimg1) / (mimg99 - mimg1)
        mimg = mscaled*255
        mimg = mimg.astype(np.uint32)

        plt.imshow(mimg,vmin=0,vmax=255)
        plt.colorbar()

        if self.cropToHoloFOV:
            plt.vlines(self.y_bounds[1],self.x_bounds[0],self.x_bounds[1],color='white')
            plt.vlines(self.y_bounds[0],self.x_bounds[0],self.x_bounds[1],color='white')
            plt.hlines(self.x_bounds[1],self.y_bounds[0],self.y_bounds[1],color='white')
            plt.hlines(self.x_bounds[0],self.y_bounds[0],self.y_bounds[1],color='white')

        H,W =reconstruction.shape
        plt.xlabel(f'FOV width {W} pix')
        plt.ylabel(f'FOV height {H} pix')
        plt.title('Reconstructed Image')
        plt.show()

    # 1
    def save_h5s(self):
        start_time_all = time.time()
        self.create_directory(self.out_path)
        self.x_slice = slice(self.x_bounds[0], self.x_bounds[1])
        self.y_slice = slice(self.y_bounds[0], self.y_bounds[1])
        for exp_name, tiff_folder in zip(self.exp_list, self.tiff_folder_list):
            pth = Path(tiff_folder)
            allMovs = list(pth.glob('*.tif'))
            print(f'found tif folder with {len(allMovs)} total tifs')
            for idx, mov in enumerate(allMovs):
                self.movs = [mov]
                self.movIndex = idx
                #print(f'found tif folder with {len(self.movs)} total tifs')
                if self.meso_params['mesoscan']:
                    if self.cropToHoloFOV:
                        self.cropToHoloFOV_save_h5_meso(exp_name,tiff_folder)
                    else:
                        self.crop_and_save_h5_meso(exp_name,tiff_folder)
                else:
                    self.crop_and_save_h5(exp_name,tiff_folder)
        elapsed_time =time.time()-start_time_all
        print(f'All saved, took {elapsed_time:.2f} seconds')

    # 2 multiplane
    def crop_and_save_h5(self,exp_name,tiff_folder):
        data = [self.load_tif(str(mov)) for mov in self.movs]
        data = np.concatenate(data)

        for plane in range(self.planes):
            #t_slice = slice(plane+self.channelOI, data.shape[0], self.planes*self.channels)
            thisoutdir = os.path.join(self.out_path, f'plane_{plane}')
            if (self.movIndex==0):
                print(thisoutdir)
                self.outdirs.append(thisoutdir)
                self.create_directory(thisoutdir)
            outname = os.path.join(thisoutdir, f'{self.movIndex}_{exp_name}_cropped_mov.h5')
            with h5py.File(outname, 'w') as hf:
                cropped_mov_plane = np.r_[tuple(data[i+self.channelOI[0]:i + len(self.channelOI), self.y_slice, self.x_slice] for i in range(plane, data.shape[0], self.planes*self.channels))]
#                 cropped_mov_plane = data[t_slice, self.y_slice, self.x_slice]
                hf.create_dataset('data', data=cropped_mov_plane, dtype='uint16')
            del cropped_mov_plane
            gc.collect()

    # 2 meso
    def crop_and_save_h5_meso(self,exp_name,tiff_folder):
        data = [self.load_tif(str(mov)) for mov in self.movs]
        numtimepointstiffile = [d.shape[0] for d in data]
        data = np.concatenate(data)
        ylen = np.shape(self.meso_params['lines'])[1]
        xlen = np.shape(self.meso_params['lines'])[0] * self.meso_params['dx'][1]
        datars = np.zeros((data.shape[0], ylen, xlen), dtype=np.int16)
        Nstripes = np.shape(self.meso_params['lines'])[0]
        for istrip in range(Nstripes):
#             t_strip = slice(istrip+self.channelOI,data.shape[0],Nstripes*self.channels)
#             datars=data[t_strip, self.meso_params['lines'][istrip], :].copy()
            datars = np.r_[tuple(data[i+self.channelOI[0]:i + len(self.channelOI), H_slice, :] for i in range(istrip, data.shape[0], Nstripes*self.channels))]
            print(f'data reshaped as {datars.shape}')
            thisoutdir = os.path.join(self.out_path, f'MROI_{istrip}')
            if (self.movIndex==0):
                print(f'saving .h5s in {thisoutdir}')
                self.outdirs.append(thisoutdir)
                self.create_directory(thisoutdir)

            outname = os.path.join(thisoutdir, f'{self.movIndex}_{exp_name}_cropped_mov.h5')
            with h5py.File(outname, 'w') as hf:
                cropped_mov_plane = datars[:, self.x_slice, self.y_slice]
                print(f'saving cropped mov of shape {cropped_mov_plane.shape} \n in {outname}')
                hf.create_dataset('data', data=cropped_mov_plane, dtype='uint16')
            del cropped_mov_plane
            gc.collect()

     # 2 meso
    def cropToHoloFOV_save_h5_meso(self,exp_name,tiff_folder):
        data = [self.load_tif(str(mov)) for mov in self.movs]
        numtimepointstiffile = [d.shape[0] for d in data]
        data = np.concatenate(data)
        ylen = np.shape(self.meso_params['lines'])[1]
        xlen = np.shape(self.meso_params['lines'])[0] * self.meso_params['dx'][1]
        datars = np.zeros((data.shape[0], ylen, xlen), dtype=np.int16)
        Nstripes = np.shape(self.meso_params['lines'])[0]
        Nstripes =self.meso_params['nrois']
        rois = []
        Nstripes = np.shape(self.meso_params['lines'])[0]
        for istrip in range(Nstripes):
            H_slice = self.meso_params['lines'][istrip]
#             t_slice = slice(istrip+self.channelOI,data.shape[0],Nstripes*self.channels)
#             datars=data[t_slice, H_slice, :].copy()
            # flexible t slicing, takes N consecutive frames where N= len(channelOI) every M frames where M= Nstripes*self.channels
            datars = np.r_[tuple(data[i+self.channelOI[0]:i + len(self.channelOI), H_slice, :] for i in range(istrip, data.shape[0], Nstripes*self.channels))]
            rois.append(datars)
        minlen= min([len(r) for r in rois])
        #
        rois = [r[:minlen,:,:] for r in rois]
        dataStack = np.concatenate(rois,axis=2)

        thisoutdir = os.path.join(self.out_path, f'HOLO_ROI_{self.x_bounds[0]}_{self.x_bounds[1]}_{ self.y_bounds[0]}_{ self.y_bounds[1]}')
        if (self.movIndex==0):
            print(f'found {len(rois)} mROIs of shape: {rois[0].shape}')
            print(f'data reshaped as {dataStack.shape}')
            print(thisoutdir)
            self.outdirs.append(thisoutdir)
            self.create_directory(thisoutdir)

        outname = os.path.join(thisoutdir, f'{self.movIndex}_{exp_name}_cropped_mov.h5')
        with h5py.File(outname, 'w') as hf:
            cropped_mov_plane = dataStack[:, self.x_slice, self.y_slice]
            #print(f'saving cropped mov of shape {cropped_mov_plane.shape},\n in {outname}')
            hf.create_dataset('data', data=cropped_mov_plane, dtype='uint16')
        del cropped_mov_plane
        gc.collect()

    # 3 helper
    def load_tif(self, mov_path):
        with ScanImageTiffReader(mov_path) as reader:
            data = reader.data()
        return data

    # 4 parallel processing
    def run_parallel_processing(self, ops):
        ops['fs'] = float(self.meso_params['fs'])/self.meso_params['nplanes']
        ops['nchannels'] =len(self.channelOI)
        print('changing some ops parameters using user info or tiff metadata \n fs: {}, nchannels: {} '.format(ops['fs'], ops['nchannels']))
        if len(self.outdirs)>1:
            ops['planes'] = 1 # means planes were already separated so each outdir has only one plane
            print('running planes in parallel')
        else:
            print(f'running single suite2p')
            ops['nplanes']= self.meso_params['nplanes'] # if all files are saved in one dir it might be multiplane data, so we take the info from the metadata of the original tif

        print('Starting parallel processing, check anaconda prompt for suite2p logs!')
        start_time_suite2p = time.time()
        db = []
        jobs = []
        for i in range(len(self.outdirs)):
            print(f'sending parallel processor number {i}')
            if i < (os.cpu_count() - 2):
                this_out_name = self.outdirs[i]+'//'
                this_db = {
                    'h5py': this_out_name,
                    'h5py_key': ['data'],
                    'look_one_level_down': True,
                    'save_path0': this_out_name,
                    'data_path': [],
                    'subfolders': [],
                    'fast_disk': []
                    }
                print(f'Will be processing this data : {this_db}')
                db.append(this_db)
                p = multiprocessing.Process(target=run_s2p, args=(ops, this_db))
                p.start()
                jobs.append(p)
            else:
                print(f'This code can only process less than {os.cpu_count() - 2} cores at the same time in this computer')
        for job in jobs:
            job.join()
        end_time_suite2p = time.time() -start_time_suite2p
        print(f'All saved, took {end_time_suite2p:.2f} seconds, bye!')
        return jobs
