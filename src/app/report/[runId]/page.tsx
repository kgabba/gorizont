import AnalyzeReport from "@/components/report/AnalyzeReport";

type Props = {
  params: Promise<{ runId: string }>;
};

export default async function ReportPage({ params }: Props) {
  const { runId } = await params;
  return <AnalyzeReport runId={runId} />;
}
