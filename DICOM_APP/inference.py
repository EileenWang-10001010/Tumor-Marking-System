from argparse import ArgumentParser
import os
import logging
import cv2
import torch
import numpy as np
import nibabel as nib
import pydicom


def ct_slices_generator( imageName, image, size=(224, 224), orientation=('L', 'A', 'S'), vgg_compatible=True,
                        scaling_value=3071):

    # Only check nii file's orientation  
    if "nii" in imageName: 

        # Make dir
        if not os.path.exists(f"./static/{imageName}_input"):
            os.makedirs(f"./static/{imageName}_input")
    
        # Load data
        scan_data = nib.load("./static/" + imageName)
        ct_scan_volume = scan_data.get_fdata() / scaling_value

        for idx in range(ct_scan_volume.shape[-1]):
            if nib.aff2axcodes(scan_data.affine) == orientation:
                original_shape = ct_scan_volume[:, :, idx].shape
                cv2.imwrite(f'./static/{imageName}_input/{idx}.jpg',(scan_data.get_fdata())[:,:,idx])
                resized_data = cv2.resize(ct_scan_volume[:, :, idx], size).astype(np.float32)
                if vgg_compatible:
                    resized_data = cv2.cvtColor(resized_data, cv2.COLOR_GRAY2RGB)
                yield np.moveaxis(resized_data, -1, 0), original_shape
            else:
                print(f"{imageName} not in desired orientation but is {nib.aff2axcodes(scan_data.affine)} instead")
    
    elif "dcm" in imageName:

        # Make dir
        if not os.path.exists(f"./static/others_input"):
            os.makedirs(f"./static/others_input")

        scan_data = pydicom.dcmread(image)
        cv2.imwrite(f'./static/others_input/{imageName}.jpg', scan_data.pixel_array)

        ct_scan_volume = scan_data.pixel_array / scaling_value

        original_shape = ct_scan_volume.shape
        resized_data = cv2.resize(ct_scan_volume, size).astype(np.float32)
        if vgg_compatible:
            resized_data = cv2.cvtColor(resized_data, cv2.COLOR_GRAY2RGB)
        yield np.moveaxis(resized_data, -1, 0), original_shape
    
    elif "jpeg" or "jpg" or "png"  in imageName:
        
        file_bytes = np.fromfile(image, np.uint32)

        try:
            # Make dir
            if not os.path.exists(f"./static/others_input"):
                os.makedirs(f"./static/others_input")

            scan_data = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            resized_data = cv2.resize(file_bytes, (512,512)).astype(np.float32) 

            cv2.imwrite(f'./static/others_input/{imageName}.jpg', resized_data)
            
            scan_data = cv2.resize(scan_data, size).astype(np.float32)
            
            ct_scan_volume = scan_data / scaling_value
            
            original_shape = ct_scan_volume.shape
            resized_data = cv2.resize(ct_scan_volume, size).astype(np.float32)

            if vgg_compatible:
                    resized_data = cv2.cvtColor(resized_data, cv2.COLOR_GRAY2RGB)
            yield np.moveaxis(resized_data, -1, 0), original_shape
        except:
            logging.info(f"Not supported file type : {imageName}")
    
    else:
        logging.info(f"Not supported file type : {imageName}")

        
'''
Inference saves a nifty file mask and masks for each slice
'''


def infer(imageName, ct_slices, model, scaling_value=3071):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    if 'nii'in imageName:

        scan_data = nib.load(f"./static/{imageName}")
        ct_scan_volume = scan_data.get_fdata() / scaling_value

    for idx, (scan_data, original_shape) in enumerate(ct_slices):
        with torch.no_grad():
            mask = model(torch.from_numpy(np.expand_dims(scan_data, axis=0)).to(device))
            resized_mask = cv2.resize(mask.squeeze(0).cpu().numpy(), original_shape, interpolation=cv2.INTER_NEAREST)
            
            if 'nii'in imageName:
                if not os.path.exists(f"./static/{imageName}_output"):
                    os.makedirs(f"./static/{imageName}_output")

                original = ct_scan_volume[:,:,idx]
                img = imgLabel(imageName,original, resized_mask)
                cv2.imwrite(f'./static/{imageName}_output/{idx}.jpg', img)
            else:
                if not os.path.exists(f"./static/others_output"):
                    os.makedirs(f"./static/others_output")

                original = cv2.imread(f"./static/others_input/{imageName}.jpg")
                
                img = imgLabel(imageName,original[:,:,0], resized_mask)

                cv2.imwrite(f'./static/others_output/{imageName}.jpg', img)
   
            

            # results.append(resized_mask)
            # np.save(os.path.join(path_to_result_dir, f"{idx}_mask"), resized_mask)
        # full_mask = np.stack(imgLabel, axis=-1)
    
    # print(full_mask.shape)

    # return full_mask

def imgLabel(imageName, original, mask):
    img = np.zeros([mask.shape[0],mask.shape[1],3])
    if 'nii' in imageName:
        img[:,:,0] = original*100*10
        img[:,:,1] = original*100*10
        img[:,:,2] = mask*100
    else:
        img[:,:,0] = original
        img[:,:,1] = original
        img[:,:,2] = mask*100
    # img[:,:,2] = 
    img = cv2.merge((img[:,:,0],img[:,:,1],img[:,:,2]))
    return img

    # print(full_mask.shape) # 512,512,1
    # if "nii" in img_path:
    #     nifti_mask = nib.Nifti1Image(full_mask, affine=np.eye(4))
    #     nib.save(nifti_mask, os.path.join(path_to_result_dir, f"label_{name}"))
    # elif "dcm" in img_path:
        # dcm = pydicom.dcmread(img_path)
        # dcm.PixelData = full_mask.tobytes()
        # dcm.save_as(os.path.join(path_to_result_dir, f"label_{name}"))
    # else:
    #     print(f" Unsupported image type : {img_path}")