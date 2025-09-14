#!/usr/bin/env python3
"""
Company Text Reader and Identifier

This script traverses all folders in output_txt directory that contain .txt annual reports,
reads the content of these reports, and uses TF-IDF and text similarity methods to identify
which companies they belong to based on the company names from company_full_list.csv.

The script ensures no folder is missed and provides detailed matching results.
"""

import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompanyTextIdentifier:
    """Main class for identifying companies in text folders"""
    
    def __init__(self, csv_path: str, output_txt_path: str):
        self.csv_path = csv_path
        self.output_txt_path = output_txt_path
        self.companies_df = None
        self.company_names = []
        self.company_codes = []
        self.folder_mappings = {}
        self.unmatched_folders = []
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True,
            strip_accents='unicode'
        )
        
    def load_companies(self) -> None:
        """Load company data from CSV file"""
        try:
            self.companies_df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.companies_df)} companies from CSV")
            
            # Extract company names and codes
            self.company_names = self.companies_df['Company'].tolist()
            self.company_codes = self.companies_df['Code'].tolist()
            
            logger.info(f"Extracted {len(self.company_names)} company names")
            
        except Exception as e:
            logger.error(f"Error loading companies from CSV: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better matching"""
        if not text:
            return ""
        
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        
        return text
    
    def extract_company_keywords(self, company_name: str) -> List[str]:
        """Extract relevant keywords from company name for matching"""
        # Remove common company suffixes and words
        stop_words = {
            'ltd', 'limited', 'corp', 'corporation', 'inc', 'incorporated', 
            'group', 'company', 'companies', 'holdings', 'holding', 'jsc',
            'joint', 'stock', 'private', 'public', 'plc', 'llc', 'co',
            'international', 'global', 'industries', 'industry', 'services',
            'solutions', 'technologies', 'technology', 'enterprises', 'enterprise'
        }
        
        # Clean company name and split into words
        cleaned_name = self.clean_text(company_name)
        words = cleaned_name.split()
        
        # Filter out stop words and short words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def read_folder_content(self, folder_path: str, max_files: int = 3) -> str:
        """Read content from text files in a folder (limited number for performance)"""
        content = ""
        txt_files = []
        
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.txt'):
                        txt_files.append(os.path.join(root, file))
        
            # Sort files and take most recent ones
            txt_files.sort(reverse=True)
            files_to_read = txt_files[:max_files]
            
            for file_path in files_to_read:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # Read first 10000 characters to avoid memory issues
                        file_content = f.read(10000)
                        content += " " + file_content
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error reading folder {folder_path}: {e}")
            
        return self.clean_text(content)
    
    def calculate_direct_similarity(self, folder_content: str, company_name: str) -> float:
        """Calculate direct text similarity between folder content and company name"""
        if not folder_content or not company_name:
            return 0.0
        
        # Check for exact company name matches
        company_keywords = self.extract_company_keywords(company_name)
        content_lower = folder_content.lower()
        
        matches = 0
        for keyword in company_keywords:
            if keyword in content_lower:
                matches += 1
        
        # Calculate similarity based on keyword matches
        if len(company_keywords) > 0:
            return matches / len(company_keywords)
        
        return 0.0
    
    def calculate_tfidf_similarity(self, folder_content: str, company_name: str) -> float:
        """Calculate TF-IDF based similarity"""
        try:
            # Prepare texts for TF-IDF
            texts = [folder_content, company_name]
            
            # Calculate TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Error calculating TF-IDF similarity: {e}")
            return 0.0
    
    def calculate_folder_name_similarity(self, folder_name: str, company_name: str, company_code: str) -> float:
        """Calculate similarity between folder name and company name/code using cosine similarity"""
        try:
            # Clean folder name and company name
            clean_folder = self.clean_text(folder_name)
            clean_company = self.clean_text(company_name)
            clean_code = self.clean_text(company_code)
            
            # Check for exact code match first
            if clean_code in clean_folder or clean_folder in clean_code:
                return 1.0
            
            # Use TF-IDF for cosine similarity between folder name and company name
            texts = [clean_folder, clean_company]
            
            if not texts[0] or not texts[1]:
                return 0.0
                
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Error calculating folder name similarity: {e}")
            return 0.0

    def find_best_match(self, folder_path: str, folder_content: str) -> Tuple[str, str, float]:
        """Find the best matching company for a folder based primarily on folder name"""
        best_match = None
        best_code = None
        best_score = 0.0
        
        folder_name = os.path.basename(folder_path)
        
        # First try exact code matching
        code_match = re.search(r'([A-Z]{2,10}(?:\.[A-Z]{2,3})?)', folder_name)
        if code_match:
            potential_code = code_match.group(1)
            
            # Check if this code exists in our company list
            for i, code in enumerate(self.company_codes):
                if code == potential_code or code.startswith(potential_code):
                    return self.company_names[i], code, 1.0
        
        # Try folder name similarity with all companies
        for i, (company_name, company_code) in enumerate(zip(self.company_names, self.company_codes)):
            # Calculate folder name similarity
            folder_sim = self.calculate_folder_name_similarity(folder_name, company_name, company_code)
            
            if folder_sim > best_score:
                best_score = folder_sim
                best_match = company_name
                best_code = company_code
        
        return best_match, best_code, best_score
    
    def get_all_folders(self) -> List[str]:
        """Get all folders containing .txt files"""
        folders = []
        
        for root, dirs, files in os.walk(self.output_txt_path):
            # Check if current directory contains .txt files
            has_txt = any(file.endswith('.txt') for file in files)
            
            if has_txt:
                folders.append(root)
            
            # Also check subdirectories
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    subdir_files = os.listdir(dir_path)
                    if any(file.endswith('.txt') for file in subdir_files):
                        folders.append(dir_path)
                except (PermissionError, FileNotFoundError):
                    continue
        
        return list(set(folders))  # Remove duplicates
    
    def process_all_folders(self) -> None:
        """Process all folders to identify companies"""
        logger.info("Getting list of all folders with .txt files...")
        all_folders = self.get_all_folders()
        
        logger.info(f"Found {len(all_folders)} folders to process")
        
        results = []
        threshold = 0.3  # Minimum similarity threshold
        
        for i, folder_path in enumerate(all_folders, 1):
            logger.info(f"Processing folder {i}/{len(all_folders)}: {folder_path}")
            
            # Read folder content
            content = self.read_folder_content(folder_path)
            
            if not content:
                logger.warning(f"No readable content found in {folder_path}")
                self.unmatched_folders.append(folder_path)
                continue
            
            # Find best match
            best_company, best_code, best_score = self.find_best_match(folder_path, content)
            
            if best_score >= threshold:
                self.folder_mappings[folder_path] = {
                    'company_name': best_company,
                    'company_code': best_code,
                    'similarity_score': best_score,
                    'folder_name': os.path.basename(folder_path)
                }
                
                results.append({
                    'folder_path': folder_path,
                    'folder_name': os.path.basename(folder_path),
                    'company_name': best_company,
                    'company_code': best_code,
                    'similarity_score': best_score,
                    'status': 'matched'
                })
                
                logger.info(f"✓ Matched: {os.path.basename(folder_path)} -> {best_company} (Score: {best_score:.3f})")
            else:
                self.unmatched_folders.append(folder_path)
                results.append({
                    'folder_path': folder_path,
                    'folder_name': os.path.basename(folder_path),
                    'company_name': best_company,
                    'company_code': best_code,
                    'similarity_score': best_score,
                    'status': 'unmatched'
                })
                
                logger.warning(f"✗ Low confidence: {os.path.basename(folder_path)} -> {best_company} (Score: {best_score:.3f})")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('/mnt/e/NEUConference/ClimateRisk/output_tfidf/company_identification_results.csv', index=False)
        
        logger.info(f"Results saved to company_identification_results.csv")
    
    def generate_summary_report(self) -> None:
        """Generate a summary report of the identification process"""
        total_folders = len(self.folder_mappings) + len(self.unmatched_folders)
        matched_folders = len(self.folder_mappings)
        unmatched_folders = len(self.unmatched_folders)
        
        summary = {
            'total_folders_processed': total_folders,
            'successfully_matched': matched_folders,
            'unmatched_folders': unmatched_folders,
            'match_rate': (matched_folders / total_folders * 100) if total_folders > 0 else 0,
            'matched_companies': list(set([info['company_name'] for info in self.folder_mappings.values()])),
            'unmatched_folder_names': [os.path.basename(path) for path in self.unmatched_folders]
        }
        
        # Save summary as JSON
        with open('/mnt/e/NEUConference/ClimateRisk/output_tfidf/identification_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("IDENTIFICATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total folders processed: {total_folders}")
        logger.info(f"Successfully matched: {matched_folders}")
        logger.info(f"Unmatched folders: {unmatched_folders}")
        logger.info(f"Match rate: {summary['match_rate']:.1f}%")
        logger.info(f"Unique companies identified: {len(summary['matched_companies'])}")
        
        if unmatched_folders > 0:
            logger.info(f"\nUnmatched folders:")
            for folder in summary['unmatched_folder_names'][:10]:  # Show first 10
                logger.info(f"  - {folder}")
            if len(summary['unmatched_folder_names']) > 10:
                logger.info(f"  ... and {len(summary['unmatched_folder_names']) - 10} more")
        
        logger.info("="*80)
    
    def run(self) -> None:
        """Run the complete identification process"""
        logger.info("Starting Company Text Identification Process")
        logger.info("="*60)
        
        # Load company data
        self.load_companies()
        
        # Process all folders
        self.process_all_folders()
        
        # Generate summary report
        self.generate_summary_report()
        
        logger.info("Process completed successfully!")


def main():
    """Main function"""
    # Define paths
    csv_path = '/mnt/e/NEUConference/ClimateRisk/output_tfidf/company_full_list.csv'
    output_txt_path = '/mnt/e/NEUConference/ClimateRisk/output_txt'
    
    # Check if paths exist
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    if not os.path.exists(output_txt_path):
        logger.error(f"Output text directory not found: {output_txt_path}")
        return
    
    # Initialize and run identifier
    identifier = CompanyTextIdentifier(csv_path, output_txt_path)
    identifier.run()


if __name__ == "__main__":
    main()
