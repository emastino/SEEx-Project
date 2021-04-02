import subprocess
import numpy as np
from pathlib import Path
import os, sys
import requests

supported_openvino_version = '2020.1'

def relative_to_abs_path(relative_path):
    dirname = Path(__file__).parent
    try:
        return str((dirname / relative_path).resolve())
    except FileNotFoundError:
        return None

model_downloader_path = relative_to_abs_path('downloader/downloader.py')
ir_converter_path     = relative_to_abs_path('downloader/converter.py')
download_folder_path  = relative_to_abs_path('downloads') + "/"


def download_model(model, model_zoo_folder):
    model_downloader_options = ['--precisions', 'FP16', '--output_dir', f'{download_folder_path}', '--cache_dir', f'{download_folder_path}/.cache', \
        '--num_attempts', '5', '--name', f'{model}', '--model_root', f'{model_zoo_folder}']

    downloader_cmd = [sys.executable, f'{model_downloader_path}']
    downloader_cmd.extend(model_downloader_options)

    # print('"{}"'.format('" "'.join(downloader_cmd)))

    result = subprocess.run(downloader_cmd)
    if result.returncode != 0:
        raise RuntimeError("Model downloader failed!")
    
    download_location = Path(download_folder_path) / model
    if(not download_location.exists()):
        raise RuntimeError(f"{download_location} doesn't exist for downloaded model!")


    return download_location



def convert_model_to_ir(model, model_zoo_folder):

    converter_path = Path(ir_converter_path)

    model_converter_options=['--precisions', 'FP16', '--output_dir', f'{download_folder_path}', '--download_dir', f'{download_folder_path}', \
        '--name', f'{model}', '--model_root', f'{model_zoo_folder}']
    converter_cmd = [sys.executable, f'{converter_path}']
    converter_cmd.extend(model_converter_options)
    
    # print('"{}"'.format('" "'.join(converter_cmd)))


    result = subprocess.run(converter_cmd)
    if result.returncode != 0:
        raise RuntimeError("Model converter failed!")
    
    ir_model_location = Path(download_folder_path) / model / "FP16"


    return ir_model_location



def myriad_compile_model_local(shaves, cmx_slices, nces, xml_path, output_file):

    myriad_compile_path = None
    if myriad_compile_path is None:
        try:
            myriad_compile_path = Path(os.environ['INTEL_OPENVINO_DIR']) / 'deployment_tools/inference_engine/lib/intel64/myriad_compile'
        except KeyError:
            sys.exit('Unable to locate Model Optimizer. '
                + 'Use --mo or run setupvars.sh/setupvars.bat from the OpenVINO toolkit.')

    PLATFORM="VPU_MYRIAD_2450" if nces == 0 else "VPU_MYRIAD_2480"

    myriad_compiler_options = ['-ip U8', '-VPU_MYRIAD_PLATFORM', f'{PLATFORM}', '-VPU_NUMBER_OF_SHAVES', f'{shaves}', \
        '-VPU_NUMBER_OF_CMX_SLICES', f'{cmx_slices}', '-m', f'{xml_path}', '-o', f'{output_file}']

    myriad_compile_cmd = np.concatenate(([myriad_compile_path], myriad_compiler_options))
    # print('"{}"'.format('" "'.join(myriad_compile_cmd)))

    result = subprocess.run(myriad_compile_cmd)
    if result.returncode != 0:
        raise RuntimeError("Myriad compiler failed!")
    
    return 0



def myriad_compile_model_cloud(xml, bin, shaves, cmx_slices, nces, output_file):
    PLATFORM="VPU_MYRIAD_2450" if nces == 0 else "VPU_MYRIAD_2480"

    # use 69.214.171 instead luxonis.com to bypass cloudflare limitation of max file size
    url = "http://69.164.214.171:8083/compile"
    payload = {
        'compile_type': 'myriad',
        'compiler_params': '-ip U8 -VPU_MYRIAD_PLATFORM ' + PLATFORM + ' -VPU_NUMBER_OF_SHAVES ' + str(shaves) +' -VPU_NUMBER_OF_CMX_SLICES ' + str(cmx_slices)
    }
    files = {
        'definition': open(Path(xml), 'rb'),
        'weights': open(Path(bin), 'rb')
    }
    params = {
        "version": supported_openvino_version
    }
    try:
        response = requests.post(url, data=payload, files=files, params=params)
        response.raise_for_status()
    except Exception as ex:
        if getattr(ex, 'response', None) is None:
            print(f"Unknown error occured: {ex}")
            return 1
        print("Model compilation failed with error code: " + str(ex.response.status_code))
        print(str(ex.response.text))
        return 2
    blob_file = open(output_file,'wb')
    blob_file.write(response.content)
    blob_file.close()

    return 0

def download_and_compile_NN_model(model, model_zoo_folder, shaves, cmx_slices, nces, output_file, model_compilation_target='auto'):

    if model_compilation_target == 'auto' or model_compilation_target == 'local':
        try:
            openvino_dir = os.environ['INTEL_OPENVINO_DIR']
            print(f'Openvino installation detected {openvino_dir}') 

            installed_openvino_version_path = Path(openvino_dir) / "deployment_tools/model_optimizer/version.txt"
            installed_openvino_version = "not detected"
            with open(installed_openvino_version_path, "r") as fp:
                installed_openvino_version = fp.read()
                installed_openvino_version = installed_openvino_version.strip()
            print("Installed openvino version: ",installed_openvino_version)
            if supported_openvino_version in installed_openvino_version:
                model_compilation_target = 'local'
                print(f'Supported openvino version installed: {supported_openvino_version}')
            else:
                if model_compilation_target == 'local':
                    raise ValueError
                model_compilation_target = 'cloud'
                print(f'Unsupported openvino version installed at {openvino_dir}, version {installed_openvino_version}, supported version is: {supported_openvino_version}')

        except:
            if model_compilation_target == 'local':
                raise SystemExit(f"Local model compilation was requested, but environment variables are not initialized for openvino {supported_openvino_version}. Run setupvars.sh/setupvars.bat from the OpenVINO toolkit.")
            model_compilation_target = 'cloud'
    
    print(f'model_compilation_target: {model_compilation_target}')
    output_location = Path(model_zoo_folder) / model / output_file

    download_location = download_model(model, model_zoo_folder)
    
    if model_compilation_target == 'local':

        ir_model_location = convert_model_to_ir(model, model_zoo_folder)

        if(not ir_model_location.exists()):
            raise RuntimeError(f"{ir_model_location} doesn't exist for downloaded model!")
        xml_path = ir_model_location / (model + ".xml")
        if(not xml_path.exists()):
            raise RuntimeError(f"{xml_path} doesn't exist for downloaded model!")
    
        return myriad_compile_model_local(shaves, cmx_slices, nces, xml_path, output_file)

    elif model_compilation_target == 'cloud':
         
        ir_model_location = Path(download_location) / "FP16"
        if(not ir_model_location.exists()):
            raise RuntimeError(f"{ir_model_location} doesn't exist for downloaded model!")
        xml_path = ir_model_location / (model + ".xml")
        if(not xml_path.exists()):
            raise RuntimeError(f"{xml_path} doesn't exist for downloaded model!")
        bin_path = ir_model_location / (model + ".bin")
        if(not bin_path.exists()):
            raise RuntimeError(f"{bin_path} doesn't exist for downloaded model!")

        result = myriad_compile_model_cloud(xml=xml_path, bin=bin_path, shaves = shaves, cmx_slices=cmx_slices, nces=nces, output_file=output_location)
        if result == 1:
            raise RuntimeError("Model compiler failed! Not connected to the internet?")
        elif result == 2:
            raise RuntimeError("Model compiler failed! Check logs for details")
    else:
        assert 'model_compilation_target must be either : ["auto", "local", "cloud"]'

    return 0

def main(args):

    model = args['model_name']
    model_zoo_folder = args['model_zoo_folder']

    shaves = args['shaves']
    cmx_slices = args['cmx_slices']
    nces =  args['nces']
    output_file = args['output']
    model_compilation_target = args['model_compilation_target']

    return download_and_compile_NN_model(model, model_zoo_folder, shaves, cmx_slices, nces, output_file, model_compilation_target)

if __name__ == '__main__':
    import argparse
    from argparse import ArgumentParser
    def parse_args():
        epilog_text = '''
        Myriad blob compiler.
        '''
        parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("-model", "--model_name", default=None,
                            type=str, required=True,
                            help="model name")
        parser.add_argument("-sh", "--shaves", default=4, type=int,
                            help="Number of shaves used by NN.")
        parser.add_argument("-cmx", "--cmx_slices", default=4, type=int,
                            help="Number of cmx slices used by NN.")
        parser.add_argument("-nce", "--nces", default=1, type=int,
                            help="Number of NCEs used by NN.")
        parser.add_argument("-o", "--output", default=None,
                            type=Path, required=True,
                            help=".blob output")
        parser.add_argument("-mct", "--model-compilation-target", default="auto",
                            type=str, required=False, choices=["auto","local","cloud"],
                            help="Compile model lcoally or in cloud?")
        parser.add_argument("-mz", "--model-zoo-folder", default=None,
                    type=str, required=True,
                    help="Path to folder with models")
        options = parser.parse_args()
        return options

    args = vars(parse_args())
    ret = main(args)
    exit(ret)