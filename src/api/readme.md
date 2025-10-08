# Arabic Sentiment Analysis API
# واجهة برمجة تطبيقات تحليل المشاعر العربية

A REST API for analyzing Arabic text sentiment using a trained BERT model. The API classifies Arabic text into three categories: **ايجابي** (Positive), **سلبي** (Negative), or **محايد** (Neutral).

## 🚀 Quick Start

pip install fastapi uvicorn pydantic

### Start the API Server

```bash
# With trained model path
python src/api/app.py --model outputs/arabic_sentiment_model/best_model --config config/sentiment_config.yaml --host 0.0.0.0 --port 8001

# With config file
python src/api/app.py --config config/sentiment_config.yaml --port 8001

# Development mode with auto-reload
python src/api/app.py --model outputs/arabic_sentiment_model/best_model --reload
```

## 🔄 Updates and Versioning

- **API Version**: 1.0.0
- **Model Version**: Check `/model/info` endpoint
- **Compatibility**: Backwards compatible within major version

## 📞 Support

For issues and questions:
1. Check this documentation
2. Review error messages in Arabic and English
3. Test with the provided client script
4. Check logs for detailed error information

---

**Made with ❤️ for Arabic NLP applications**
