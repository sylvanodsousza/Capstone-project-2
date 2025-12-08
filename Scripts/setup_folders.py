import os

def setup_folders():
    print("📁 Setting up project folder structure...\n")

    folders = [
        "outputs",
        "outputs/processed",
        "outputs/plots",
        "outputs/tables",
        "scripts",
        "data"
    ]

    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"✔ Created / Verified: {folder}")
        except Exception as e:
            print(f"❌ Error creating {folder}: {e}")

    print("\n🎉 Folder structure set up successfully!")
    print("You can now run clean.py and all scripts without directory errors.")

if __name__ == "__main__":
    setup_folders()
