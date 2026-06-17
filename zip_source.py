import zipfile
import os

zip_name = r"c:\Users\KIIT\Desktop\flipkartgrid\source_files.zip"
files_to_zip = [
    r"c:\Users\KIIT\Desktop\flipkartgrid\predict.py",
    r"c:\Users\KIIT\Desktop\flipkartgrid\README.txt",
    r"c:\Users\KIIT\Desktop\flipkartgrid\README.md",
    r"c:\Users\KIIT\Desktop\flipkartgrid\PROJECT_REPORT.md",
    r"c:\Users\KIIT\Desktop\flipkartgrid\Traffic_Demand_Prediction.ipynb",
    r"c:\Users\KIIT\Desktop\flipkartgrid\submission.csv",
    r"c:\Users\KIIT\Desktop\flipkartgrid\index.html",
    r"c:\Users\KIIT\Desktop\flipkartgrid\styles.css",
    r"c:\Users\KIIT\Desktop\flipkartgrid\app.js",
    r"c:\Users\KIIT\Desktop\flipkartgrid\prepare_dashboard_data.py"
]

print("Creating ZIP file...")
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
    for file_path in files_to_zip:
        if os.path.exists(file_path):
            basename = os.path.basename(file_path)
            zip_ref.write(file_path, arcname=basename)
            print(f"Added {basename} to zip")
        else:
            print(f"File {file_path} does not exist!")

print("ZIP file created successfully at:", zip_name)
print("ZIP file size (bytes):", os.path.getsize(zip_name))
