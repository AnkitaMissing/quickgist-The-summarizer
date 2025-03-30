from fpdf import FPDF
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_user_description():
    """Get text description from user"""
    print("\nEnter a description for your clustering analysis (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "" and len(lines) > 0:  # Stop on double Enter
            break
        if line:
            lines.append(line)
    return "\n".join(lines)

def generate_clusters():
    """Generate sample data and perform clustering"""
    np.random.seed(42)
    X = np.random.rand(100, 2)
    kmeans = KMeans(n_clusters=3).fit(X)
    labels = kmeans.predict(X)
    return X, labels

def create_visualization(X, labels):
    """Create and save cluster visualization"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title("K-Means Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Cluster')
    plot_filename = "clusters.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    return plot_filename

def create_pdf_report(description, plot_filename):
    """Generate PDF report with user description and visualization"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Machine Learning Analysis Report", ln=1, align='C')
    pdf.ln(10)
    
    # User description section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="User Description:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, txt=description)
    pdf.ln(10)
    
    # Visualization section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Cluster Visualization:", ln=1)
    pdf.image(plot_filename, x=10, y=None, w=180)
    
    # Footer
    pdf.set_y(-15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Generated using K-Means clustering", align='C')
    
    output_filename = "ml_analysis_report.pdf"
    pdf.output(output_filename)
    return output_filename

def main():
    print("=== K-Means Clustering Analysis Report Generator ===")
    
    # Get user input
    user_description = get_user_description()
    
    # Generate clusters and visualization
    X, labels = generate_clusters()
    plot_file = create_visualization(X, labels)
    
    # Create PDF report
    output_file = create_pdf_report(user_description, plot_file)
    
    print(f"\nPDF report successfully generated: {output_file}")

if __name__ == "__main__":
    main()