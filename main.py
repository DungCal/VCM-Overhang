import time
import multiprocessing as mp

import Core.config as config
import Core.file_utils as file_utils
import Core.image_processing as image_processing


def main():
    start_time = time.time()

    templates = file_utils.load_templates(config.TEMPLATE_FOLDER)
    if not templates:
        print("No templates.")
        return

    output_folders = file_utils.create_output_folders()

    image_files = [f for f in config.INPUT_FOLDER.glob(
        "*") if f.suffix.lower() in config.ALLOWED_EXTENSIONS]
    total_images = len(image_files)
    if total_images == 0:
        print("No images found.")
        return
    tasks = [(path, output_folders) for path in image_files]

    num_processes = config.NUM_PROCESSES
    if config.DEBUG:
        print(f"Using {num_processes} worker processes.")

    with mp.Pool(processes=num_processes, initializer=image_processing.init_worker, initargs=(templates,)) as pool:
        results = pool.starmap(image_processing.process_image_pipeline, tasks)

    end_time = time.time()
    ok_count = results.count("OK")
    ng_count = results.count("NG")

    print(f"Total time: {end_time - start_time:.2f} seconds.")
    print(f"Processed images: {total_images}")
    print(f"  OK images: {ok_count}")
    print(f"  NG images: {ng_count}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
