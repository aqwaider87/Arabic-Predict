#!/usr/bin/env python3
"""
Arabic Sentiment Classification - Main Application
نظام تصنيف المشاعر العربية

Usage:
    python main.py create-config
    python main.py train --config config/sentiment_config.yaml
    python main.py predict --model outputs/arabic_sentiment_model/best_model --text "النص العربي هنا"
    python main.py predict --model outputs/arabic_sentiment_model/best_model --file input.csv --output results.csv
"""

import argparse
import sys
from pathlib import Path
import logging
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_config(args):
    """Create a new configuration file"""
    try:
        from config import create_default_config_file
        
        config_path = getattr(args, 'config', 'config/sentiment_config.yaml')
        create_default_config_file(config_path)
        
        print(f"✅ Configuration file created: {config_path}")
        print(f"💡 Edit the file to customize your settings")
        print(f"🚀 Then run: python main.py train")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Config creation failed: {e}")
        return False

def validate_config(args):
    """Validate configuration file"""
    try:
        from config import load_config
        
        config_path = getattr(args, 'config', 'config/sentiment_config.yaml')
        config = load_config(config_path)
        
        print(f"✅ Configuration file is valid: {config_path}")
        print(f"📊 Configuration summary:")
        print(f"   Data file: {config['data']['csv_path']}")
        print(f"   Text column: {config['data']['text_column']}")
        print(f"   Label column: {config['data']['label_column']}")
        print(f"   Output dir: {config['output_dir']}")
        print(f"   Model: {config['model']['pretrained_name']}")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Config validation failed: {e}")
        return False

def train_model(args):
    """Train a new model"""
    try:
        from trainer import quick_train
        from config import get_config
        
        logger.info("🚀 Starting training...")
        
        # Load configuration
        config_path = getattr(args, 'config', 'config/sentiment_config.yaml')
        config = get_config(config_path)
        
        # Override with command line arguments
        if hasattr(args, 'data') and args.data:
            config['data']['csv_path'] = args.data
        if hasattr(args, 'output') and args.output:
            config['output_dir'] = args.output
        
        logger.info(f"⚙️ Using config: {config_path}")
        logger.info(f"📊 Data file: {config['data']['csv_path']}")
        logger.info(f"📁 Output directory: {config['output_dir']}")
        
        # Train model
        trained_model, test_results = quick_train(
            config=config,
            resume=not getattr(args, 'no_resume', False)
        )
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📈 Test Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
        logger.info(f"📈 Test F1-Macro: {test_results.get('eval_f1_macro', 0):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        return False

def predict_text(args):
    """Predict sentiment for single text"""
    try:
        from predictor import ArabicSentimentPredictor
        
        # Load model
        predictor = ArabicSentimentPredictor(args.model)
        
        # Predict
        result = predictor.predict_single(args.text)
        
        # Display results
        print("\n" + "="*60)
        print("🔮 نتيجة تصنيف المشاعر")
        print("="*60)
        
        if result["success"]:
            print(f"📝 النص: {result['text']}")
            print(f"🎯 التصنيف: {result['prediction']}")
            print(f"📊 الثقة: {result['confidence']:.3f}")
            print(f"🔍 الطريقة: {result['method']}")
            
            if result.get("emojis_found"):
                print(f"😊 الإيموجي: {', '.join(result['emojis_found'])}")
            
            if result.get("probabilities"):
                print("\n📈 توزيع الاحتماليات:")
                for label, prob in sorted(result["probabilities"].items(), 
                                        key=lambda x: x[1], reverse=True):
                    bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                    print(f"  {label:8}: {prob:.3f} |{bar}|")
        else:
            print(f"❌ خطأ: {result['error']}")
        
        print("="*60)
        return True
        
    except Exception as e:
        logger.error(f"💥 Prediction failed: {e}")
        return False

def predict_file(args):
    """Predict sentiment for CSV file"""
    try:
        from predictor import ArabicSentimentPredictor
        
        # Load model
        logger.info(f"🔍 Loading model from {args.model}")
        predictor = ArabicSentimentPredictor(args.model)
        
        # Load input file
        logger.info(f"📂 Loading input file: {args.file}")
        try:
            df = pd.read_csv(args.file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(args.file, encoding='cp1256')
        
        # Determine text column
        text_col = getattr(args, 'text_column', None) or 'text'
        if text_col not in df.columns:
            # Use first column if 'text' not found
            text_col = df.columns[0]
            logger.info(f"📝 Using column '{text_col}' as text column")
        
        # Predict
        logger.info(f"🔮 Predicting sentiment for {len(df)} texts...")
        results_df = predictor.predict_dataframe(df, text_col)
        
        # Save results
        output_file = getattr(args, 'output', None) or args.file.replace('.csv', '_predictions.csv')
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Print summary
        successful = results_df['sentiment_prediction'].notna().sum()
        total = len(results_df)
        
        print(f"\n📊 ملخص النتائج:")
        print(f"📝 إجمالي النصوص: {total}")
        print(f"✅ تنبؤات ناجحة: {successful}")
        print(f"❌ تنبؤات فاشلة: {total - successful}")
        print(f"📈 معدل النجاح: {successful/total*100:.1f}%")
        
        if successful > 0:
            sentiment_counts = results_df['sentiment_prediction'].value_counts()
            print(f"\n🎭 توزيع المشاعر:")
            for sentiment, count in sentiment_counts.items():
                percentage = (count / successful) * 100
                print(f"  {sentiment:8}: {count:4} ({percentage:5.1f}%)")
        
        print(f"\n💾 تم حفظ النتائج في: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"💥 File prediction failed: {e}")
        return False

def debug_data(args):
    """Debug and analyze data file"""
    try:
        from data_loader import DataLoader
        from config import get_config
        
        # Load configuration or use default
        config_path = getattr(args, 'config', 'config/sentiment_config.yaml')
        try:
            config = get_config(config_path)
            data_file = args.data or config['data']['csv_path']
            text_col = config['data']['text_column']
            label_col = config['data']['label_column']
        except:
            # Use provided data file with defaults
            data_file = args.data
            text_col = 'text'
            label_col = 'sentiment'
        
        logger.info(f"🔍 Analyzing data file: {data_file}")
        
        # Load and analyze data
        data_loader = DataLoader(config if 'config' in locals() else {})
        
        # Load file
        df = data_loader.load_csv(data_file)
        print(f"\n📊 ملف البيانات: {data_file}")
        print(f"📏 عدد الصفوف: {len(df)}")
        print(f"📋 عدد الأعمدة: {len(df.columns)}")
        print(f"🏷️ الأعمدة: {list(df.columns)}")
        
        # Show sample data
        print(f"\n📝 عينة من البيانات:")
        print(df.head())
        
        # Check for common text columns
        text_candidates = ['text', 'content', 'message', 'review', 'comment']
        label_candidates = ['label', 'sentiment', 'class', 'category']
        
        print(f"\n🔍 احتمالات أعمدة النص:")
        for col in df.columns:
            if any(candidate in col.lower() for candidate in text_candidates):
                print(f"  ✅ {col}")
        
        print(f"\n🔍 احتمالات أعمدة التصنيف:")
        for col in df.columns:
            if any(candidate in col.lower() for candidate in label_candidates):
                print(f"  ✅ {col}")
                # Show unique values
                unique_vals = df[col].unique()[:10]  # First 10 unique values
                print(f"     القيم: {unique_vals}")
        
        # Test validation on sample
        from validator import ArabicValidator
        validator = ArabicValidator()
        
        print(f"\n🧪 اختبار التحقق من النصوص:")
        sample_texts = df.iloc[:5, 0].tolist()  # First 5 texts from first column
        
        for i, text in enumerate(sample_texts):
            print(f"\n  {i+1}. النص: {repr(text)}")
            is_valid, cleaned, error = validator.validate_text(str(text))
            print(f"     صحيح: {is_valid}")
            if not is_valid:
                print(f"     الخطأ: {error}")
            else:
                print(f"     منظف: {repr(cleaned)}")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Debug failed: {e}")
        return False

def show_status(args):
    """Show training status and available models"""
    try:
        output_dir = Path(getattr(args, 'output', 'outputs'))
        
        print(f"\n📊 حالة المشروع: {output_dir}")
        print("=" * 50)
        
        # Check for trained models
        best_model = output_dir / "arabic_sentiment_model" / "best_model"
        if not best_model.exists():
            best_model = output_dir / "best_model"
        
        if best_model.exists():
            print(f"🏆 أفضل نموذج: {best_model}")
            
            # Load model info
            info_file = best_model / "model_info.json"
            if info_file.exists():
                import json
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                print(f"📊 أفضل نتيجة F1: {info.get('best_metric', 'غير متاح')}")
                print(f"🕐 تاريخ الانتهاء: {info.get('training_completed', 'غير متاح')}")
        else:
            print("❌ لا يوجد نموذج مُدرب")
        
        # Check for checkpoints
        checkpoints = list(output_dir.glob("**/checkpoint-*"))
        if checkpoints:
            print(f"\n📁 نقاط الحفظ المتاحة: {len(checkpoints)}")
            latest = max(checkpoints, key=lambda x: int(x.name.split('-')[-1]))
            print(f"🔄 آخر checkpoint: {latest}")
        else:
            print("\n❌ لا توجد نقاط حفظ")
        
        # Check for config file
        config_file = Path('config/sentiment_config.yaml')
        if config_file.exists():
            print(f"\n📋 ملف التكوين: {config_file}")
        else:
            print(f"\n⚠️ ملف التكوين غير موجود")
            print(f"💡 قم بإنشاؤه: python main.py create-config")
        
        print("=" * 50)
        return True
        
    except Exception as e:
        logger.error(f"💥 Status check failed: {e}")
        return False

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Arabic Sentiment Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create configuration file
  python main.py create-config

  # Train a model
  python main.py train

  # Predict single text
  python main.py predict --model outputs/arabic_sentiment_model/best_model --text "هذا المنتج ممتاز"

  # Predict from file
  python main.py predict --model outputs/arabic_sentiment_model/best_model --file input.csv

  # Debug data
  python main.py debug --data data.csv

  # Show status
  python main.py status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create config command
    create_config_parser = subparsers.add_parser('create-config', help='Create configuration file')
    create_config_parser.add_argument('--config', default='config/sentiment_config.yaml', help='Configuration file path')
    
    # Validate config command
    validate_config_parser = subparsers.add_parser('validate-config', help='Validate configuration file')
    validate_config_parser.add_argument('--config', default='config/sentiment_config.yaml', help='Configuration file path')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--config', default='config/sentiment_config.yaml', help='Configuration file path')
    train_parser.add_argument('--data', help='Path to training CSV file (overrides config)')
    train_parser.add_argument('--output', help='Output directory (overrides config)')
    train_parser.add_argument('--no-resume', action='store_true', help='Do not resume from checkpoint')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict sentiment')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    
    # Predict subcommands
    predict_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_group.add_argument('--text', help='Single text to predict')
    predict_group.add_argument('--file', help='CSV file to predict')
    
    # Optional arguments for file prediction
    predict_parser.add_argument('--output', help='Output file for predictions')
    predict_parser.add_argument('--text-column', help='Name of text column in CSV')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Debug and analyze data')
    debug_parser.add_argument('--data', required=True, help='Path to CSV file to analyze')
    debug_parser.add_argument('--config', default='config/sentiment_config.yaml', help='Configuration file path')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show project status')
    status_parser.add_argument('--output', default='outputs', help='Project directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return False
    
    # Execute command
    success = False
    
    try:
        if args.command == 'create-config':
            success = create_config(args)
        elif args.command == 'validate-config':
            success = validate_config(args)
        elif args.command == 'train':
            success = train_model(args)
        elif args.command == 'predict':
            if args.text:
                success = predict_text(args)
            elif args.file:
                success = predict_file(args)
        elif args.command == 'debug':
            success = debug_data(args)
        elif args.command == 'status':
            success = show_status(args)
        else:
            parser.print_help()
            success = False
            
    except KeyboardInterrupt:
        logger.info("⏹️ تم إيقاف البرنامج بواسطة المستخدم")
        success = False
    except Exception as e:
        logger.error(f"💥 خطأ غير متوقع: {e}")
        success = False
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 خطأ فادح: {e}")
        sys.exit(1)