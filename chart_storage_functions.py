# chart_storage_functions.py
import os
import json
import pickle
from datetime import datetime

def get_chart_folder(app_name):
    """Get chart folder path for specific app"""
    chart_folder = f"charts/{app_name}"
    if not os.path.exists(chart_folder):
        os.makedirs(chart_folder, exist_ok=True)
    return chart_folder

def save_chart_to_folder(app_name, chart_data, chart_analysis):
    """Save chart data and analysis to folder"""
    try:
        chart_folder = get_chart_folder(app_name)
        
        # Save chart data
        chart_data_path = os.path.join(chart_folder, "chart_data.json")
        with open(chart_data_path, 'w', encoding='utf-8') as f:
            json.dump(chart_data, f, ensure_ascii=False, indent=2)
        
        # Save chart analysis
        chart_analysis_path = os.path.join(chart_folder, "chart_analysis.json")
        with open(chart_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(chart_analysis, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata = {
            'app_name': app_name,
            'created_at': datetime.now().isoformat(),
            'chart_data_size': len(str(chart_data)),
            'analysis_sections': list(chart_analysis.keys())
        }
        metadata_path = os.path.join(chart_folder, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Chart data saved successfully for {app_name}")
        return True
        
    except Exception as e:
        print(f"Error saving chart data for {app_name}: {e}")
        return False

def load_chart_from_folder(app_name):
    """Load chart data and analysis from folder"""
    try:
        chart_folder = get_chart_folder(app_name)
        
        # Check if files exist
        chart_data_path = os.path.join(chart_folder, "chart_data.json")
        chart_analysis_path = os.path.join(chart_folder, "chart_analysis.json")
        
        if not (os.path.exists(chart_data_path) and os.path.exists(chart_analysis_path)):
            return None, None
        
        # Load chart data
        with open(chart_data_path, 'r', encoding='utf-8') as f:
            chart_data = json.load(f)
        
        # Load chart analysis
        with open(chart_analysis_path, 'r', encoding='utf-8') as f:
            chart_analysis = json.load(f)
        
        print(f"Chart data loaded successfully for {app_name}")
        return chart_data, chart_analysis
        
    except Exception as e:
        print(f"Error loading chart data for {app_name}: {e}")
        return None, None

def chart_exists_in_folder(app_name):
    """Check if chart data exists in folder"""
    chart_folder = get_chart_folder(app_name)
    chart_data_path = os.path.join(chart_folder, "chart_data.json")
    chart_analysis_path = os.path.join(chart_folder, "chart_analysis.json")
    
    return os.path.exists(chart_data_path) and os.path.exists(chart_analysis_path)

def delete_chart_from_folder(app_name):
    """Delete chart data from folder"""
    try:
        chart_folder = get_chart_folder(app_name)
        
        files_to_delete = ["chart_data.json", "chart_analysis.json", "metadata.json"]
        deleted_files = []
        
        for filename in files_to_delete:
            file_path = os.path.join(chart_folder, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(filename)
        
        # Try to remove empty directory
        try:
            if os.path.exists(chart_folder) and not os.listdir(chart_folder):
                os.rmdir(chart_folder)
        except OSError:
            pass  # Directory not empty or other error
        
        print(f"Deleted chart files for {app_name}: {deleted_files}")
        return True
        
    except Exception as e:
        print(f"Error deleting chart data for {app_name}: {e}")
        return False

def get_chart_metadata(app_name):
    """Get chart metadata if exists"""
    try:
        chart_folder = get_chart_folder(app_name)
        metadata_path = os.path.join(chart_folder, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
        
    except Exception as e:
        print(f"Error getting metadata for {app_name}: {e}")
        return None