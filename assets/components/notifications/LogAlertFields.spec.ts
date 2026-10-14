/**
 * @vitest-environment jsdom
 */
import { mount } from "@vue/test-utils";
import { describe, expect, test, vi } from "vitest";
import { createI18n } from "vue-i18n";
import type { NotificationRule } from "@/types/notifications";

const config = vi.hoisted(() => ({ base: "", hosts: [], aiEnabled: false }));

vi.mock("@/stores/config", () => ({
  __esModule: true,
  default: config,
  withBase: (path: string) => path,
}));

import LogAlertFields from "./LogAlertFields.vue";

const i18n = createI18n({ legacy: false, locale: "en", missingWarn: false, fallbackWarn: false, messages: { en: {} } });

function mountFields(alert?: Partial<NotificationRule>) {
  return mount(LogAlertFields, {
    props: {
      alert: alert as NotificationRule | undefined,
      containerExpression: "",
      hasContainers: false,
      isLoading: false,
      validatePreview: vi.fn(async () => ({ data: null })),
    },
    global: { plugins: [i18n], stubs: { ExpressionField: true, LogList: true } },
  });
}

describe("<LogAlertFields /> AI toggle", () => {
  test("is hidden without an AI key, and the stored value is still sent", () => {
    config.aiEnabled = false;
    const wrapper = mountFields({ logExpression: 'level == "error"', aiSummary: true });

    expect(wrapper.find('input[type="checkbox"]').exists()).toBe(false);
    expect(wrapper.vm.typeFields.aiSummary).toBe(true);
  });

  test("is off by default and sent once ticked", async () => {
    config.aiEnabled = true;
    const wrapper = mountFields();

    const toggle = wrapper.find('input[type="checkbox"]');
    expect(wrapper.vm.typeFields.aiSummary).toBe(false);
    await toggle.setValue(true);
    expect(wrapper.vm.typeFields.aiSummary).toBe(true);
  });
});
