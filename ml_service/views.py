from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .explainer import get_explainer
from .predictor import get_predictor


class PredictFreshnessView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            data = request.data
            predictor = get_predictor()
            explainer = get_explainer()

            result = predictor.predict(data)
            shap_features = explainer.explain(data, top_n=3)

            return Response(
                {
                    "success": True,
                    "prediction": result.to_dict(),
                    "shap_explanation": shap_features,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"success": False, "error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class ExplainView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        try:
            data = request.data
            explainer = get_explainer()

            shap_features = explainer.explain(data, top_n=5)

            return Response(
                {
                    "success": True,
                    "shap_explanation": shap_features,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"success": False, "error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
