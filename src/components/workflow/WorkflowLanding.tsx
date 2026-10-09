import SiteHeader from "@/components/SiteHeader";
import UploadSection from "@/components/UploadSection";
import { EXAMPLE_REPORT_PATH } from "@/lib/exampleReport";
import { WORKFLOW_STEPS, type WorkflowStep } from "./steps";

const PRODUCT_STEP =
  WORKFLOW_STEPS.find((s) => s.isProduct) ?? WORKFLOW_STEPS[4];

export default function WorkflowLanding() {
  return (
    <div className="workflow-shell">
      <SiteHeader />

      <div className="workflow-grid">
        <aside className="workflow-rail" aria-label="Цепочка оценки">
          <ol className="workflow-list">
            {WORKFLOW_STEPS.map((step, i) => (
              <li
                key={step.id}
                className={`workflow-list-item${step.isProduct ? " is-product" : ""}`}
              >
                {i > 0 ? (
                  <div className="workflow-connector" aria-hidden>
                    <span className="workflow-connector-shaft" />
                    <span className="workflow-connector-head" />
                  </div>
                ) : null}
                <StepRow step={step} />
              </li>
            ))}
          </ol>
        </aside>

        <section className="workflow-panel">
          <DetailPanel step={PRODUCT_STEP} />
        </section>
      </div>

      <UploadSection />

      <footer id="contacts" className="site-contacts">
        <div className="site-contacts-inner">
          <p className="site-contacts-label">Контакты</p>
          <div className="site-contacts-links">
            <a href="mailto:geo.discovery@mail.ru" className="site-contacts-mail">
              geo.discovery@mail.ru
            </a>
            <a href="tel:+79167982259" className="site-contacts-phone">
              +7 916 798-22-59
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

function StepRow({ step }: { step: WorkflowStep }) {
  const Icon = step.Icon;
  return (
    <div
      className={`workflow-step${step.isProduct ? " is-product" : ""}`}
    >
      <span className="workflow-step-icon">
        <Icon />
      </span>
      <span className="workflow-step-body">
        {step.isProduct && step.productBadge ? (
          <span className="workflow-step-badge">{step.productBadge}</span>
        ) : null}
        <span className="workflow-step-title">{step.title}</span>
      </span>
    </div>
  );
}

function DetailPanel({ step }: { step: WorkflowStep }) {
  return (
    <div className={`workflow-detail${step.isProduct ? " is-product" : ""}`}>
      <h1 className="workflow-detail-title">{step.panelTitle}</h1>
      <div className="workflow-detail-rule" aria-hidden />
      <p className="workflow-detail-lead">{step.panelLead}</p>

      {step.subSteps?.length ? (
        <ol className="workflow-sublist">
          {step.subSteps.map((sub) => {
            const SubIcon = sub.Icon;
            return (
              <li key={sub.id} className="workflow-subitem">
                <span className="workflow-sub-num">{sub.num}</span>
                <span className="workflow-sub-icon">
                  <SubIcon />
                </span>
                <span className="workflow-sub-title">{sub.title}</span>
              </li>
            );
          })}
        </ol>
      ) : null}

      {step.output ? (
        <div className="workflow-output">
          <p className="workflow-output-label">Выход</p>
          <p className="workflow-output-text">{step.output}</p>
        </div>
      ) : null}

      {step.cta ? (
        <div className="workflow-detail-actions">
          <a href="#analyze" className="workflow-cta">
            {step.cta.label}
          </a>
          <a href={EXAMPLE_REPORT_PATH} className="workflow-cta-secondary">
            Пример отчёта
          </a>
        </div>
      ) : null}
    </div>
  );
}
